import argparse
import logging
import os
import pprint
import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from copy import deepcopy

from dataset.semi_chn6 import SemiDataset
from model.semseg.dpt import DPT
from supervised import evaluate
from util.classes import CLASSES
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
from util.losses import *

parser = argparse.ArgumentParser(description='Semi-supervised training for SSGW')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

        writer = SummaryWriter(args.save_path)

        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass']})
    state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth')
    model.backbone.load_state_dict(state_dict, strict=False)

    if cfg['lock_backbone']:
        model.lock_backbone()

    optimizer = AdamW(
        [
            {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg['lr']},
            {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
             'lr': cfg['lr'] * cfg['lr_multi']}
        ],
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01
    )

    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=True)
    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False

    if cfg['criterion']['name'] != 'CELoss':
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])
    criterion_l = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=255),
                            DiceLoss(smooth=0.05, ignore_index=255), 1.0, 1.0).cuda(local_rank)
    criterion_dice = DiceLoss(mode='multiclass', ignore_index=255).cuda(local_rank)

    trainset_u = SemiDataset(cfg, cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    
    trainset_l = SemiDataset(cfg, cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg, cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best, previous_best_ema = 0.0, 0.0
    best_epoch, best_epoch_ema = 0, 0
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model_ema.load_state_dict(checkpoint['model_ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        previous_best_ema = checkpoint['previous_best_ema']
        best_epoch = checkpoint['best_epoch']
        best_epoch_ema = checkpoint['best_epoch_ema']

        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)


    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}, '
                        'EMA: {:.2f} @epoch-{:}'.format(epoch, previous_best, best_epoch, previous_best_ema,
                                                        best_epoch_ema))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_u_h = AverageMeter()
        total_loss_u_v = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)
        model.train()
        for i, (((img_x, mask_x), img_a_x),
                (img_u_w, img_u_a_w, img_u_s1, img_u_a_s1, img_u_s2, img_u_a_s2, _ignore_mask)) in enumerate(loader):


            img_x, mask_x, img_a_x = img_x.cuda(), mask_x.cuda(),img_a_x.cuda()
            img_u_w, img_u_a_w, img_u_s1, img_u_a_s1, img_u_s2, img_u_a_s2 = img_u_w.cuda(), img_u_a_w.cuda(),\
                                                     img_u_s1.cuda(), img_u_a_s1.cuda(), \
                                                        img_u_s2.cuda(), img_u_a_s2.cuda()

            with torch.no_grad():
                pred_u_w = model_ema(img_u_w, a=img_u_a_w).detach()
                mask_u_w = pred_u_w.argmax(dim=1)
                # ACGM weight map from pseudo label
                eta_u = build_acgm_eta(
                    mask_u_w,
                    epoch=epoch,
                    max_epoch=cfg['epochs'],
                    kernel_size=63,
                    strides=(1, 2, 4)
                )

            out = model(img_x, a=img_a_x)
            out_u_h, out_u_v = model(torch.cat((img_u_s1, img_u_s2)), a=torch.cat((img_u_a_s1, img_u_a_s2)), comp_drop=True).chunk(2)
            loss_x = criterion_l(out, mask_x)

            loss_u_h = weighted_ce_loss(out_u_h, mask_u_w, eta_u) + criterion_dice(out_u_h, mask_u_w)
            loss_u_v = weighted_ce_loss(out_u_v, mask_u_w, eta_u) + criterion_dice(out_u_v, mask_u_w)

            # 根据配置组合损失
            if epoch <= 5:
                loss = loss_x
            else:
                loss = loss_x + (loss_u_h + loss_u_v) / 2.0  # 基础损失

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_u_h.update(loss_u_h.item())
            total_loss_u_v.update(loss_u_v.item())

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Synchronize loss across all processes
            torch.distributed.all_reduce(loss)
            loss = loss / world_size

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            ema_ratio = min(1 - 1 / (iters + 1), 0.996)

            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))

            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_u_h', loss_u_h.item(), iters)
                writer.add_scalar('train/loss_u_v', loss_u_v.item(), iters)

            if (i % max(1, len(trainloader_l) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, '
                            'Loss u_h: {:.3f}, Loss u_v: {:.3f}'.format(
                    i, total_loss.avg, total_loss_x.avg,
                    total_loss_u_h.avg, total_loss_u_v.avg))
        eval_mode = 'sliding_window'  if cfg['dataset'] == 'mass' else 'original'
        # Before evaluation
        torch.distributed.barrier()
        mIoU, iou_class, _, _, _, _ = evaluate(model, valloader, eval_mode, cfg, multiplier=14 if cfg['model'] == 'dpt' else None)
        mIoU_ema, iou_class_ema, _, _, _, _ = evaluate(model_ema, valloader, eval_mode, cfg, multiplier=14 if cfg['model'] == 'dpt' else None)

        # Add synchronization point
        torch.distributed.barrier()

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}, '
                            'EMA: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou,
                                                 iou_class_ema[cls_idx]))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}, EMA: {:.2f}\n'.format(eval_mode, mIoU, mIoU_ema))

            writer.add_scalar('eval/mIoU', mIoU, epoch)
            writer.add_scalar('eval/mIoU_ema', mIoU_ema, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)
                writer.add_scalar('eval/%s_IoU_ema' % (CLASSES[cfg['dataset']][i]), iou_class_ema[i], epoch)

        is_best = mIoU_ema >= previous_best_ema
        previous_best = max(mIoU, previous_best)
        previous_best_ema = max(mIoU_ema, previous_best_ema)

        if mIoU == previous_best:
            best_epoch = epoch
        if mIoU_ema == previous_best_ema:
            best_epoch_ema = epoch
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'model_ema': model_ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
                'previous_best_ema': previous_best_ema,
                'best_epoch': best_epoch,
                'best_epoch_ema': best_epoch_ema
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
