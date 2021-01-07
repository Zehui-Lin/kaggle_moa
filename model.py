'''
Author: Zehui Lin
Date: 2021-01-04 22:04:45
LastEditors: Zehui Lin
LastEditTime: 2021-01-05 19:24:34
Description: file content
'''
import torch
import torch.nn as nn


class TabularNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(len(cfg.num_features), cfg.hidden_size),
            nn.BatchNorm1d(cfg.hidden_size),
            nn.Dropout(cfg.dropout),
            nn.PReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.BatchNorm1d(cfg.hidden_size),
            nn.Dropout(cfg.dropout),
            nn.PReLU(),
            nn.Linear(cfg.hidden_size, len(cfg.target_cols)),
        )

    def forward(self, cont_x, cate_x):
        # no use of cate_x yet
        x = self.mlp(cont_x)
        return x


class TabularNNV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(len(cfg.num_features)+len(cfg.cat_features), cfg.hidden_size),
            nn.BatchNorm1d(cfg.hidden_size),
            nn.Dropout(cfg.dropout),
            nn.PReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.BatchNorm1d(cfg.hidden_size),
            nn.Dropout(cfg.dropout),
            nn.PReLU(),
            nn.Linear(cfg.hidden_size, len(cfg.target_cols)),
        )

    def forward(self, cont_x, cate_x):
        # use of cate_x yet
        x = self.mlp(torch.cat((cont_x, cate_x), dim=1))
        return x
