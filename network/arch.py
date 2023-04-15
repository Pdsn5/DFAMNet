import torch
import torch_scatter
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from network.basic_block import Lovasz_loss
from network.spvcnn import get_model as SPVCNN
from network.base_model import LightningBaseModel
from network.basic_block import ResNetFCN
class Attention(nn.Module):
    def __init__(self, channels):
        super(Attention, self).__init__()
        self.pseudo_in, self.valid_in = channels
        middle = self.valid_in // 4
        self.fc1 = nn.Linear(self.pseudo_in, middle)
        self.fc2 = nn.Linear(self.valid_in, middle)
        self.fc3 = nn.Linear(2*middle, 2)
        self.conv1 = nn.Sequential(nn.Conv1d(self.pseudo_in, self.valid_in, 1),
                                    nn.BatchNorm1d(self.valid_in),
                                    nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(self.valid_in, self.valid_in, 1),
                                    nn.BatchNorm1d(self.valid_in),
                                    nn.ReLU())

    def forward(self, pseudo_feas, valid_feas):
        # pfeas = []
        # ifeas = []
        # leng = 0
        # for i in range(len(list)):
        #     if i == 0:
        #         pfeas.append(pseudo_feas[leng:(leng + list[i])])
        #         ifeas.append(valid_feas[leng:(leng + list[i])])
        #         leng += list[i]
        #     else:
        #         pfeas.append(pseudo_feas[leng:(leng + list[i])])
        #         ifeas.append(valid_feas[leng:(leng + list[i])])
        #         leng += list[i]
        pseudo_feas = torch.unsqueeze(pseudo_feas, dim=0).permute(0,2,1)
        valid_feas = torch.unsqueeze(valid_feas, dim=0).permute(0,2,1)
        batch = pseudo_feas.size(0)

        pseudo_feas_f = pseudo_feas.transpose(1,2).contiguous().view(-1, self.pseudo_in)
        valid_feas_f = valid_feas.transpose(1,2).contiguous().view(-1, self.valid_in)

        pseudo_feas_f_ = self.fc1(pseudo_feas_f)
        valid_feas_f_ = self.fc2(valid_feas_f)
        pseudo_valid_feas_f = torch.cat([pseudo_feas_f_, valid_feas_f_],dim=-1)
        weight = torch.sigmoid(self.fc3(pseudo_valid_feas_f))

        pseudo_weight = weight[:,0].squeeze()
        pseudo_weight = pseudo_weight.view(batch, 1, -1)

        valid_weight = weight[:,1].squeeze()
        valid_weight = valid_weight.view(batch, 1, -1)

        pseudo_features_att = self.conv1(pseudo_feas)  * pseudo_weight
        valid_features_att     =  self.conv2(valid_feas)      *  valid_weight
        pseudo_features_att = pseudo_features_att.squeeze(dim=0).permute(1, 0)
        valid_features_att = valid_features_att.squeeze(dim=0).permute(1, 0)
        features_att =pseudo_features_att + valid_features_att
        return features_att
class xModalKD(nn.Module):
    def __init__(self,config):
        super(xModalKD, self).__init__()
        self.hiden_size = config['model_params']['hiden_size']
        self.scale_list = config['model_params']['scale_list']
        self.num_classes = config['model_params']['num_classes']
        self.lambda_xm = config['train_params']['lambda_xm']
        self.lambda_seg2d = config['train_params']['lambda_seg2d']
        self.num_scales = len(self.scale_list)

        self.multihead_3d_classifier = nn.ModuleList()
        for i in range(self.num_scales):
            self.multihead_3d_classifier.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size, 128),
                    nn.ReLU(True),
                    nn.Linear(128, self.num_classes))
            )

        self.multihead_fuse_classifier = nn.ModuleList()
        for i in range(self.num_scales):
            self.multihead_fuse_classifier.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size, 128),
                    nn.ReLU(True),
                    nn.Linear(128, self.num_classes))
            )
        self.leaners = nn.ModuleList()
        self.fcs1 = nn.ModuleList()
        self.fcs2 = nn.ModuleList()
        for i in range(self.num_scales):
            self.leaners.append(nn.Sequential(nn.Linear(self.hiden_size, self.hiden_size)))
            self.fcs1.append(nn.Sequential(nn.Linear(self.hiden_size * 2, self.hiden_size)))
            self.fcs2.append(nn.Sequential(nn.Linear(self.hiden_size, self.hiden_size)))

        self.classifier = nn.Sequential(
            nn.Linear(self.hiden_size * self.num_scales, 128),
            nn.ReLU(True),
            nn.Linear(128, self.num_classes),
        )

        if 'seg_labelweights' in config['dataset_params']:
            seg_num_per_class = config['dataset_params']['seg_labelweights']
            seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
            seg_labelweights = torch.Tensor(np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0))
        else:
            seg_labelweights = None
        self.attention = Attention(channels=[64, 64])
        self.ce_loss = nn.CrossEntropyLoss(weight=seg_labelweights, ignore_index=config['dataset_params']['ignore_label'])
        self.lovasz_loss = Lovasz_loss(ignore=config['dataset_params']['ignore_label'])

    @staticmethod
    def p2img_mapping(pts_fea, p2img_idx, batch_idx):
        img_feat = []
        for b in range(batch_idx.max()+1):
            img_feat.append(pts_fea[batch_idx == b][p2img_idx[b]])
        return torch.cat(img_feat, 0)

    @staticmethod
    def voxelize_labels(labels, full_coors):
        lbxyz = torch.cat([labels.reshape(-1, 1), full_coors], dim=-1)
        unq_lbxyz, count = torch.unique(lbxyz, return_counts=True, dim=0)
        inv_ind = torch.unique(unq_lbxyz[:, 1:], return_inverse=True, dim=0)[1]
        label_ind = torch_scatter.scatter_max(count, inv_ind)[1]
        labels = unq_lbxyz[:, 0][label_ind]
        return labels

    def seg_loss(self, logits, labels):
        ce_loss = self.ce_loss(logits, labels)
        lovasz_loss = self.lovasz_loss(F.softmax(logits, dim=1), labels)
        return ce_loss + lovasz_loss

    def forward(self, data_dict):
        loss = 0
        img_seg_feat = []
        batch_idx = data_dict['batch_idx']
        point2img_index = data_dict['point2img_index']

        for idx, scale in enumerate(self.scale_list):
            img_feat = data_dict['img_scale{}'.format(scale)]
            pts_feat = data_dict['layer_{}'.format(idx)]['pts_feat']
            coors_inv = data_dict['scale_{}'.format(scale)]['coors_inv']

            # 3D prediction
            pts_pred_full = self.multihead_3d_classifier[idx](pts_feat)

            # correspondence
            pts_label_full = self.voxelize_labels(data_dict['labels'], data_dict['layer_{}'.format(idx)]['full_coors'])
            pts_feat = self.p2img_mapping(pts_feat[coors_inv], point2img_index, batch_idx)
            pts_pred = self.p2img_mapping(pts_pred_full[coors_inv], point2img_index, batch_idx)

            fuse_pred = self.attention(pts_feat, img_feat)


            img_seg_feat.append(fuse_pred)
            fuse_pred = self.multihead_fuse_classifier[idx](fuse_pred)

            # Segmentation Loss
            seg_loss_3d = self.seg_loss(pts_pred_full, pts_label_full)
            seg_loss_2d = self.seg_loss(fuse_pred, data_dict['img_label'])
            loss += seg_loss_3d + seg_loss_2d * self.lambda_seg2d / self.num_scales

            # KL divergence
            xm_loss = F.kl_div(
                F.log_softmax(pts_pred, dim=1),
                F.softmax(fuse_pred, dim=1),
            )
            loss += xm_loss * self.lambda_xm / self.num_scales

        img_seg_logits = self.classifier(torch.cat(img_seg_feat, 1))
        loss += self.seg_loss(img_seg_logits, data_dict['img_label'])
        data_dict['loss'] += loss

        return data_dict


class get_model(LightningBaseModel):
    def __init__(self, config):
        super(get_model, self).__init__(config)
        self.save_hyperparameters()
        self.baseline_only = config.baseline_only
        self.num_classes = config.model_params.num_classes
        self.hiden_size = config.model_params.hiden_size
        self.lambda_seg2d = config.train_params.lambda_seg2d
        self.lambda_xm = config.train_params.lambda_xm
        self.scale_list = config.model_params.scale_list
        self.num_scales = len(self.scale_list)

        self.model_3d = SPVCNN(config)
        if not self.baseline_only:
            self.model_2d = ResNetFCN(
                backbone=config.model_params.backbone_2d,
                pretrained=config.model_params.pretrained2d,
                config=config
            )
            self.fusion = xModalKD(config)
        else:
            print('Start vanilla training!')

    def forward(self, data_dict):
        # 3D network
        data_dict = self.model_3d(data_dict)

        # training with 2D network
        if self.training and not self.baseline_only:
            data_dict = self.model_2d(data_dict)
            data_dict = self.fusion(data_dict)

        return data_dict
