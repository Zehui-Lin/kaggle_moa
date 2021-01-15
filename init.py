'''
Author: Zehui Lin
Date: 2021-01-05 10:33:24
LastEditors: Zehui Lin
LastEditTime: 2021-01-15 16:58:17
Description: file content
'''


class CFG_class:
    ex_name = 'add_cate_x_ema'
    num_fold = 5
    max_grad_norm = 1000
    gradient_accumulation_steps = 1
    hidden_size = 512
    dropout = 0.5
    lr = 1e-2
    weight_decay = 1e-6
    batch_size = 6400
    epochs = 20
    # total_cate_size=5
    # emb_size=4

    def __init__(self, num_features=None, cat_features=None, target_cols=None):
        CFG_class.num_features = num_features
        CFG_class.cat_features = cat_features
        CFG_class.target_cols = target_cols
