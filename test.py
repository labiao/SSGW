import argparse
import logging
import os

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.semi_chn6 import SemiDataset
from model.semseg.dpt import DPT
from supervised import evaluate
from util.classes import CLASSES
from util.utils import init_log, count_params
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description='Single-GPU evaluation for SSGW road segmentation')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--model-path', type=str, required=True, help='Path to best.pth')
    parser.add_argument('--gpu', default='0', type=str, help='GPU ID to use')
    parser.add_argument(
        '--checkpoint-key',
        default='auto',
        choices=['auto', 'model', 'model_ema'],
        help='State-dict key. auto prefers model_ema for semi-supervised checkpoints.',
    )
    args = parser.parse_args()

    # 1. 设置环境变量和日志
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logger = init_log('global', logging.INFO)
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger.info(f'Using GPU: {args.gpu}')
    logger.info(f'Loading config from: {args.config}')

    # 2. 构建模型
    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass']})
    logger.info('Total params: {:.1f}M'.format(count_params(model)))

    # 3. 加载模型权重
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"No checkpoint found at {args.model_path}")

    checkpoint = torch.load(args.model_path, map_location='cpu')

    checkpoint_key = args.checkpoint_key
    if checkpoint_key == 'auto':
        checkpoint_key = 'model_ema' if 'model_ema' in checkpoint else 'model'
    if checkpoint_key not in checkpoint:
        raise KeyError(
            f"Checkpoint key '{checkpoint_key}' not found. "
            f"Available keys: {sorted(checkpoint.keys())}"
        )
    logger.info(f'Loading checkpoint state: {checkpoint_key}')

    # 处理 DDP 训练产生的 'module.' 前缀
    state_dict = checkpoint[checkpoint_key]
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    # 4. 准备测试数据集 (单卡模式下无需 Sampler)
    testset = SemiDataset(cfg, cfg['dataset'], cfg['data_root'], 'val')
    testloader = DataLoader(
        testset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
        drop_last=False
    )

    # 5. 执行评估
    eval_mode = 'sliding_window' if cfg['dataset'] == 'mass' else 'original'
    logger.info(f"Starting evaluation mode: {eval_mode}")

    multiplier = 14
    # 直接调用你的 evaluate 函数
    with torch.no_grad():
        mIoU, iou_class, F1, recall_class, precision_class, f1_class = evaluate(model, testloader, eval_mode, cfg, multiplier=multiplier, dst=True)
        # for img, id in tqdm(testloader):
        for img, img_a, _, id in tqdm(testloader):
            img, img_a = img.cuda(), img_a.cuda()
            # img = img.cuda()
            if eval_mode == 'original':
                if multiplier is not None:
                    ori_h, ori_w = img.shape[-2:]
                    if multiplier == 512:
                        new_h, new_w = 512, 512
                    else:
                        new_h, new_w = int(ori_h / multiplier + 0.5) * multiplier, int(
                            ori_w / multiplier + 0.5) * multiplier
                    img = F.interpolate(img, (new_h, new_w), mode='bilinear', align_corners=True)
                    img_a = F.interpolate(img_a, (new_h, new_w), mode='bilinear', align_corners=True)

                pred = model(img, img_a)
                # pred = model(img)

                if multiplier is not None:
                    pred = F.interpolate(pred, (ori_h, ori_w), mode='bilinear', align_corners=True)
                pred = pred.argmax(dim=1)
                for i in range(pred.shape[0]):
                    # 只取当前 i 的预测，不覆盖 pred 变量
                    single_pred = pred[i].squeeze().cpu().numpy().astype(np.uint8)
                    save_path = os.path.join(os.path.dirname(args.model_path) + '/pred', id[i].split('/')[-1])
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    ext = os.path.splitext(save_path)[1] or ".png"
                    ok, encoded = cv2.imencode(ext, single_pred * 255)
                    if not ok:
                        raise RuntimeError(f"Failed to encode prediction image: {save_path}")
                    encoded.tofile(save_path)

    # 6. 打印结果
    # 4. 打印格式化结果
    logger.info("=" * 85)
    logger.info(f"{'Class':<15} | {'IoU':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
    logger.info("-" * 85)
    for i in range(cfg['nclass']):
        logger.info(
            f"{CLASSES[cfg['dataset']][i]:<15} | {iou_class[i]:8.2f} | {precision_class[i]:10.2f} | {recall_class[i]:10.2f} | {f1_class[i]:10.2f}")

    logger.info('-' * 85)
    logger.info(f'Mean IoU: {mIoU:.2f}')
    logger.info(f'F1 Score: {F1:.2f}')
    logger.info('=' * 85)


if __name__ == '__main__':
    main()
