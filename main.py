'''
Author: Zehui Lin
Date: 2021-01-04 20:42:51
LastEditors: Zehui Lin
LastEditTime: 2021-01-05 19:24:27
Description: file content
'''
import torch
import numpy as np
import pandas as pd
from init import CFG_class
from run_train import run_kfold_nn
from sklearn.metrics import log_loss
from utils import get_logger, seed_everything, cate2num
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# Gpu Setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Utils
logger = get_logger()
seed_everything(seed=7)

# Data Loading
train_features = pd.read_csv('data/train_features.csv')
train_targets_scored = pd.read_csv('data/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv('data/train_targets_nonscored.csv')
test_features = pd.read_csv('data/test_features.csv')
submission = pd.read_csv('data/sample_submission.csv')

train = train_features.merge(train_targets_scored, on='sig_id')
target_cols = [c for c in train_targets_scored.columns if c not in ['sig_id']]

# cols = target_cols + ['cp_type']
# train[cols].groupby('cp_type').sum().sum(1) # ctl_vehicle = 0

'''tl_vehicle 所有的moa标签都为0'''
train = train[train['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
test = test_features[test_features['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)


# Dataset
cat_features = ['cp_time', 'cp_dose']
num_features = [c for c in train.columns if train.dtypes[c] != 'object']
num_features = [c for c in num_features if c not in cat_features]  # 额外重新编码
num_features = [c for c in num_features if c not in target_cols]  # 去除标签列
target = train[target_cols].values
train = cate2num(train)
test = cate2num(test)

# Training Configuration
CFG = CFG_class(num_features, cat_features, target_cols)

# CV split 增加一列、记录交叉验证
# folds比train多一列
folds = train.copy()
Fold = MultilabelStratifiedKFold(n_splits=CFG_class.num_fold, shuffle=True, random_state=7)
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[target_cols])):
    folds.loc[val_index, 'fold'] = int(n)
folds['fold'] = folds['fold'].astype(int)

# Seed Averaging for solid result
oof = np.zeros((len(train), len(CFG.target_cols)))
predictions = np.zeros((len(test), len(CFG.target_cols)))

# 3次5折交叉验证平均
SEED = [7, 77, 777]
for seed in SEED:
    _oof, _predictions = run_kfold_nn(CFG,
                                      train, test, folds,
                                      num_features, cat_features, target,
                                      device, logger,
                                      n_fold=CFG_class.num_fold, seed=seed)
    oof += _oof / len(SEED)
    predictions += _predictions / len(SEED)

score = 0
for i in range(target.shape[1]):
    _score = log_loss(target[:, i], oof[:, i])
    score += _score / target.shape[1]
logger.info(f"Seed Averaged CV score: {score}")

train[target_cols] = oof
train[['sig_id']+target_cols].to_csv('data/oof.csv', index=False)

test = test.merge(train_targets_scored, on='sig_id')
test[target_cols] = predictions
test[['sig_id']+target_cols].to_csv('data/pred.csv', index=False)


# Final result with 'cp_type'=='ctl_vehicle' data
result = train_targets_scored.drop(columns=target_cols)\
    .merge(train[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)
y_true = train_targets_scored[target_cols].values
y_pred = result[target_cols].values
score = 0
for i in range(y_true.shape[1]):
    _score = log_loss(y_true[:, i], y_pred[:, i])
    score += _score / y_true.shape[1]
logger.info(f"Final result: {score}")

# Submit
sub = submission.drop(columns=target_cols).merge(test[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)
sub.to_csv('data/submission.csv', index=False)
sub.head()
