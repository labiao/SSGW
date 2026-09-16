import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.dinov2 import DINOv2
from model.util.blocks import FeatureFusionBlock, _make_scratch


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )

class StyleExtractor(nn.Module):
    def __init__(self, em_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, em_dim, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(em_dim, em_dim, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(em_dim, em_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, images_a):
        x = self.relu(self.conv1(images_a))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.avgpool(x)                  # [B, C, 1, 1]
        x = x.flatten(2).transpose(1, 2)     # [B, 1, C]
        return x

class DPTHead(nn.Module):
    def __init__(
        self, 
        nclass,
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024],
    ):
        super(DPTHead, self).__init__()
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        # 新增：tuned feature 融合
        self.tune_proj4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[3], kernel_size=1, stride=1, padding=0),
            nn.ReLU(True)
        )
        self.tune_proj3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[2], kernel_size=1, stride=1, padding=0),
            nn.ReLU(True)
        )

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features, nclass, kernel_size=1, stride=1, padding=0)
        )
    
    def forward(self, out_features, patch_h, patch_w, tune_feature=None):
        out = []
        for i, x in enumerate(out_features):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out

        # ===== 新增：融合 tuned feature =====
        if tune_feature is not None:
            tune4 = F.interpolate(tune_feature[1], size=layer_4.shape[2:], mode='bilinear', align_corners=True)
            tune3 = F.interpolate(tune_feature[0], size=layer_3.shape[2:], mode='bilinear', align_corners=True)

            layer_4 = layer_4 + self.tune_proj4(tune4)
            layer_3 = layer_3 + self.tune_proj3(tune3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv(path_1)
        
        return out


class DPT(nn.Module):
    def __init__(
        self, 
        encoder_size='base', 
        nclass=2,
        features=128, 
        out_channels=[96, 192, 384, 768], 
        use_bn=False,
    ):
        super(DPT, self).__init__()

        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11], 
            'large': [4, 11, 17, 23], 
            'giant': [9, 19, 29, 39]
        }

        self.frozen_indices = self.intermediate_layer_idx[encoder_size]
        self.tuned_indices = [7, 10]
        self.split_start = self.tuned_indices[0] - 1  # 8
        
        self.encoder_size = encoder_size
        self.backbone = DINOv2(model_name=encoder_size,
                               use_adapter=True,
                               adapter_indexes=self.intermediate_layer_idx[encoder_size],  # 或先只用后3层
                               adapter_num_heads=1,
                               adapter_ratio=1.0,
                               )
        self.style_extractor = StyleExtractor(self.backbone.embed_dim)
        self.head = DPTHead(nclass, self.backbone.embed_dim, features, use_bn, out_channels=out_channels)
        
        self.binomial = torch.distributions.binomial.Binomial(probs=0.5)

        # ===== 新增：split head =====
        # 这里假设你的 self.backbone 内部能访问 vit blocks
        vit = self.backbone.model if hasattr(self.backbone, "model") else self.backbone

        split_blocks = [copy.deepcopy(vit.blocks[i]) for i in self.tuned_indices]
        self.split_head = nn.ModuleList(split_blocks)

        self.split_norm = copy.deepcopy(vit.norm)

    def tokens_to_map(self, x, patch_h, patch_w):
        # x: [B, N, C] or [B, 1+N, C]
        if x.shape[1] == patch_h * patch_w + 1:
            x = x[:, 1:, :]
        x = self.split_norm(x)
        x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)
        return x

    def lock_backbone(self):
        for name, p in self.backbone.named_parameters():
            trainable = ("style_injection" in name
                    or ".adapter." in name
            )
            p.requires_grad = trainable
    
    def forward(self, x, a=None, comp_drop=False):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        a = self.style_extractor(a)
        features = self.backbone.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder_size],
            a=a
        )
        features = tuple(features)
        # 3) tuned branch: 从 layer8 的输出继续走 [9,10,11]
        tuned_features = []
        for idx, blk in enumerate(self.split_head):
            tuned_features.append(blk(features[idx + 2], a=None))
        if comp_drop:
            bs, dim = features[0].shape[0], features[0].shape[-1]
            
            dropout_mask1 = self.binomial.sample((bs // 2, dim)).cuda() * 2.0
            dropout_mask2 = 2.0 - dropout_mask1
            dropout_prob = 0.5
            num_kept = int(bs // 2 * (1 - dropout_prob))
            kept_indexes = torch.randperm(bs // 2)[:num_kept]
            dropout_mask1[kept_indexes, :] = 1.0
            dropout_mask2[kept_indexes, :] = 1.0
            
            dropout_mask = torch.cat((dropout_mask1, dropout_mask2))
            
            features = (feature * dropout_mask.unsqueeze(1) for feature in features)
            tuned_features = (feature * dropout_mask.unsqueeze(1) for feature in tuned_features)

        tuned_features = [self.tokens_to_map(tuned_feature, patch_h, patch_w) for tuned_feature in tuned_features]
        
        out = self.head(features, patch_h, patch_w, tune_feature=tuned_features)
        out = F.interpolate(out, (patch_h * 14, patch_w * 14), mode='bilinear', align_corners=True)
        
        return out
