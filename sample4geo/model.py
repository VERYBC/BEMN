import torch
import timm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sample4geo.FDA import FDA

class TimmModel(nn.Module):

    def __init__(self,
                 model_name,
                 pretrained = True,
                 first = False,
                 second = False
                 ):

        super(TimmModel, self).__init__()

        self.first = first
        self.second = second
        self.conv_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.dino_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.conv_compress = torch.nn.AdaptiveAvgPool1d(12)
        self.dino_compress = torch.nn.AdaptiveAvgPool1d(784)

        if self.first:
            self.projection1 = nn.Linear(144, 400)
            self.projection2 = nn.Linear(784, 400)

        if self.second:
            self.fda = FDA()

        if "Tiny" in model_name:
            self.model1 = timm.create_model('convnextv2_nano.fcmae_ft_in22k_in1k_384', pretrained=pretrained, num_classes=0)
            self.model2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        elif "Small" in model_name:
            self.model1 = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k_384', pretrained=pretrained, num_classes=0)
            self.model2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        elif "Base" in model_name:
            self.model1 = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k_384', pretrained=pretrained, num_classes=0)
            self.model2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_config(self, ):
        data_config1 = timm.data.resolve_model_data_config(self.model1)
        return data_config1

    def set_grad_checkpointing(self, enable=True):
        self.model1.set_grad_checkpointing(enable)

    def forward(self, img1, img2=None):

        if img2 is not None:

            if self.first:
                # ------------------------------------------------------------img1
                # convnext
                x = self.model1.stem(img1)
                feat1_1 = self.model1.stages(x)
                local_feat1 = self.conv_pool(feat1_1).squeeze(-2, -1)

                c1 = local_feat1

                # dino
                x = self.model2.prepare_tokens_with_masks(img1)
                for blk in self.model2.blocks:
                    x = blk(x)
                x_norm = self.model2.norm(x)
                ret = {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1: self.model2.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.model2.num_register_tokens + 1:],
                    "x_prenorm": x,
                }
                feat1_2 = self.model2.head(ret["x_norm_patchtokens"])
                global_feat1 = self.dino_pool(feat1_2.permute(0, 2, 1)).squeeze(-1)

                d1 = global_feat1

                # JFA
                # # local branch
                B, C, H, W = feat1_1.shape[0], feat1_1.shape[1], feat1_1.shape[2], feat1_1.shape[3]
                feat1_1 = self.conv_compress(feat1_1.reshape(B, C * H, W)).view(B, C, -1)
                feat1_1 = self.projection1(feat1_1)
                feat1_1 = F.normalize(feat1_1, p=2, dim=-1)

                #  # global branch
                feat1_2 = self.dino_compress(feat1_2.permute(0, 2, 1))
                feat1_2 = self.projection2(feat1_2)
                feat1_2 = F.normalize(feat1_2, p=2, dim=-1)

                feat_alignment1_1 = torch.cat((feat1_1, feat1_2), dim=1)

                # # joint
                B, C, L = feat1_2.shape
                feat1_1 = feat1_1.unsqueeze(1)
                feat1_1 = F.interpolate(feat1_1, size=(C, L), mode='bilinear', align_corners=True)
                feat1_1 = feat1_1.squeeze(1)
                feat1_2 = feat1_2

                feat_alignment1_2 = feat1_1 * feat1_2
                feat_alignment1_2 = F.normalize(feat_alignment1_2, p=2, dim=-1)

                feat_alignment1 = torch.cat((feat_alignment1_1, feat_alignment1_2), dim=1)

                # ------------------------------------------------------------img2
                # convnext
                x = self.model1.stem(img2)
                feat2_1 = self.model1.stages(x)
                local_feat2 = self.conv_pool(feat2_1).squeeze(-2, -1)

                c2 = local_feat2

                # dino
                x = self.model2.prepare_tokens_with_masks(img2)
                for blk in self.model2.blocks:
                    x = blk(x)
                x_norm = self.model2.norm(x)
                ret = {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1: self.model2.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.model2.num_register_tokens + 1:],
                    "x_prenorm": x,
                }
                feat2_2 = self.model2.head(ret["x_norm_patchtokens"])
                global_feat2 = self.dino_pool(feat2_2.permute(0, 2, 1)).squeeze(-1)

                d2 = global_feat2

                # JFA
                # # local branch
                feat2_1 = feat2_1.flatten(2)
                feat2_1 = self.projection1(feat2_1)
                feat2_1 = F.normalize(feat2_1, p=2, dim=-1)

                # # global branch
                feat2_2 = self.projection2(feat2_2.permute(0, 2, 1))
                feat2_2 = F.normalize(feat2_2, p=2, dim=-1)

                feat_alignment2_1 = torch.cat((feat2_1, feat2_2), dim=1)

                # # joint
                B, C, L = feat2_2.shape
                feat2_1 = feat2_1.unsqueeze(1)  # (B, 1, 384, 256)
                feat2_1 = F.interpolate(feat2_1, size=(C, L), mode='bilinear', align_corners=True)
                feat2_1 = feat2_1.squeeze(1)
                feat2_2 = feat2_2

                feat_alignment2_2 = feat2_1 * feat2_2
                feat_alignment2_2 = F.normalize(feat_alignment2_2, p=2, dim=-1)

                feat_alignment2 = torch.cat((feat_alignment2_1, feat_alignment2_2), dim=1)

                return feat_alignment1, feat_alignment2, c1, c2, d1, d2

            elif self.second:
                # FDA
                weight_feature1, frequency_feat1 = self.fda(img1)

                weights1 = torch.sigmoid(weight_feature1)
                local_weights1 = weights1[:, 0].unsqueeze(-1)
                global_weights1 = 1 - local_weights1

                weight_feature2, frequency_feat2 = self.fre(img2)

                weights2 = torch.sigmoid(weight_feature2)
                local_weights2 = weights2[:, 0].unsqueeze(-1)
                global_weights2 = 1 - local_weights2

                # ------------------------------------------------------------img1
                # convnext
                x = self.model1.stem(img1)
                feat1_1 = self.model1.stages(x)
                local_feat1 = self.conv_pool(feat1_1).squeeze(-2, -1)

                # dino
                x = self.model2.prepare_tokens_with_masks(img1)
                for blk in self.model2.blocks:
                    x = blk(x)
                x_norm = self.model2.norm(x)
                ret = {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1: self.model2.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.model2.num_register_tokens + 1:],
                    "x_prenorm": x,
                }
                feat1_2 = self.model2.head(ret["x_norm_patchtokens"])
                global_feat1 = self.dino_pool(feat1_2.permute(0, 2, 1)).squeeze(-1)

                # fusion
                local_feat1 = F.normalize(local_feat1, p=2, dim=-1)
                global_feat1 = F.normalize(global_feat1, p=2, dim=-1)

                feat1 = torch.cat((local_feat1 * local_weights1, global_feat1 * global_weights1), dim=-1)

                # ------------------------------------------------------------img2
                # convnext
                x = self.model1.stem(img2)
                feat2_1 = self.model1.stages(x)
                local_feat2 = self.conv_pool(feat2_1).squeeze(-2, -1)

                # dino
                x = self.model2.prepare_tokens_with_masks(img2)
                for blk in self.model2.blocks:
                    x = blk(x)
                x_norm = self.model2.norm(x)
                ret = {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1: self.model2.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.model2.num_register_tokens + 1:],
                    "x_prenorm": x,
                }
                feat2_2 = self.model2.head(ret["x_norm_patchtokens"])
                global_feat2 = self.dino_pool(feat2_2.permute(0, 2, 1)).squeeze(-1)

                # fusion
                local_feat2 = F.normalize(local_feat2, p=2, dim=-1)
                global_feat2 = F.normalize(global_feat2, p=2, dim=-1)

                feat2 = torch.cat((local_feat2 * local_weights2, global_feat2 * global_weights2), dim=-1)

                return feat1, feat2, frequency_feat1, frequency_feat2

        else:
            # convnext
            x = self.model1.stem(img1)
            feat1 = self.model1.stages(x)
            local_feat = self.conv_pool(feat1).squeeze(-2, -1)

            # dino
            x = self.model2.prepare_tokens_with_masks(img1)
            for blk in self.model2.blocks:
                x = blk(x)
            x_norm = self.model2.norm(x)
            ret = {
                "x_norm_clstoken": x_norm[:, 0],
                "x_norm_regtokens": x_norm[:, 1: self.model2.num_register_tokens + 1],
                "x_norm_patchtokens": x_norm[:, self.model2.num_register_tokens + 1:],
                "x_prenorm": x,
            }
            feat2 = self.model2.head(ret["x_norm_patchtokens"])
            global_feat = self.dino_pool(feat2.permute(0, 2, 1)).squeeze(-1)

            # fusion
            local_feat = F.normalize(local_feat, p=2, dim=-1)
            global_feat = F.normalize(global_feat, p=2, dim=-1)

            feat = torch.cat((local_feat, global_feat), dim=-1)

            return feat

