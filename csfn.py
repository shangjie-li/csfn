import torch.nn as nn
import torch.nn.functional as F


def build_model(cfg, num_features):
    if cfg['type'] == 'CSFN':
        return CSFN(cfg=cfg, num_features=num_features)
    else:
        raise NotImplementedError


class CSSR(nn.Module):
    def __init__(self, num_features, filters):
        super().__init__()

        filters = [num_features] + filters
        self.layers = nn.ModuleList()
        for i in range(len(filters) - 1):
            cur_layers = nn.Sequential(
                nn.Conv2d(filters[i], filters[i + 1], kernel_size=1),
                nn.ReLU()
            )
            self.layers.append(cur_layers)
        self.layers.append(
            nn.Conv2d(filters[-1], 1, kernel_size=1)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x  # [batch_size, 1, 1, max_objs]


class CSFN(nn.Module):
    def __init__(self, cfg, num_features):
        super().__init__()

        self.cssr1 = CSSR(num_features, cfg['filters'])
        self.cssr2 = CSSR(num_features, cfg['filters'])

    def forward(self, batch_dict):
        batch_size = batch_dict['batch_size']
        mask1 = batch_dict['mask1'].bool()  # [batch_size, max_objs]
        mask2 = batch_dict['mask2'].bool()  # [batch_size, max_objs]

        x1 = self.cssr1(batch_dict['x1'])
        x2 = self.cssr2(batch_dict['x2'])
        x1 = x1.view(batch_size, -1)  # [batch_size, max_objs]
        x2 = x2.view(batch_size, -1)  # [batch_size, max_objs]

        if self.training:
            y1 = batch_dict['y1']  # [batch_size, max_objs]
            y2 = batch_dict['y2']  # [batch_size, max_objs]
            loss1 = F.l1_loss(x1[mask1], y1[mask1]) / max(mask1.sum(), 1e-5)
            loss2 = F.l1_loss(x2[mask2], y2[mask2]) / max(mask2.sum(), 1e-5)

            total_loss = loss1 + loss2
            stats_dict = {
                'cssr1': loss1.item(),
                'cssr2': loss2.item(),
            }

            return total_loss, stats_dict

        else:
            batch_scores1 = []
            batch_scores2 = []

            for batch_idx in range(batch_size):
                scores1 = x1[batch_idx][mask1[batch_idx]]
                scores2 = x2[batch_idx][mask2[batch_idx]]
                batch_scores1.append(scores1)
                batch_scores2.append(scores2)

            batch_dict['pred_scores1'] = batch_scores1  # list of tensor
            batch_dict['pred_scores2'] = batch_scores2  # list of tensor

            return batch_dict
