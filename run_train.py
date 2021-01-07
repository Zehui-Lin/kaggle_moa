'''
Author: Zehui Lin
Date: 2021-01-04 22:30:44
LastEditors: Zehui Lin
LastEditTime: 2021-01-05 18:28:36
Description: file content
'''
import os
import torch
import numpy as np
import pandas as pd
from model import TabularNN, TabularNNV2
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader
from utils import get_logger, seed_everything
from dataset import TrainDataset, TestDataset
from epoch_fun import train_fn, validate_fn, inference_fn


def run_single_nn(cfg, train, test, folds, num_features, cat_features, target, device, logger, fold_num=0, seed=7):

    # Set seed
    logger.info(f'Set seed {seed}')
    seed_everything(seed=seed)

    # loader
    trn_idx = folds[folds['fold'] != fold_num].index
    val_idx = folds[folds['fold'] == fold_num].index
    train_folds = train.loc[trn_idx].reset_index(drop=True)
    valid_folds = train.loc[val_idx].reset_index(drop=True)
    train_target = target[trn_idx]
    valid_target = target[val_idx]
    train_dataset = TrainDataset(train_folds, num_features, cat_features, train_target)
    valid_dataset = TrainDataset(valid_folds, num_features, cat_features, valid_target)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, drop_last=False)

    # model
    if cfg.ex_name == "baseline":
        model = TabularNN(cfg)
    if cfg.ex_name == "add_cate_x":
        model = TabularNNV2(cfg)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
                                                    max_lr=1e-2, epochs=cfg.epochs, steps_per_epoch=len(train_loader))

    # log
    log_df = pd.DataFrame(columns=(['EPOCH']+['TRAIN_LOSS']+['VALID_LOSS']))

    # train & validate
    best_loss = np.inf
    for epoch in range(cfg.epochs):
        train_loss = train_fn(train_loader, model, optimizer, scheduler, device)
        valid_loss, val_preds = validate_fn(valid_loader, model, device)
        log_row = {'EPOCH': epoch,
                   'TRAIN_LOSS': train_loss,
                   'VALID_LOSS': valid_loss,
                   }
        log_df = log_df.append(pd.DataFrame(log_row, index=[0]), sort=False)
        # logger.info(log_df.tail(1))
        if valid_loss < best_loss:
            logger.info(f'epoch{epoch} save best model... {valid_loss}')
            best_loss = valid_loss
            oof = np.zeros((len(train), len(cfg.target_cols)))
            oof[val_idx] = val_preds
            torch.save(model.state_dict(), os.path.join(cfg.ex_name, f"fold{fold_num}_seed{seed}.pth"))

    # predictions
    test_dataset = TestDataset(test, num_features, cat_features)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    if cfg.ex_name == "baseline":
        model = TabularNN(cfg)
    if cfg.ex_name == "add_cate_x":
        model = TabularNNV2(cfg)
    model.load_state_dict(torch.load(os.path.join(cfg.ex_name, f"fold{fold_num}_seed{seed}.pth")))
    model.to(device)
    predictions = inference_fn(test_loader, model, device)

    # del
    torch.cuda.empty_cache()

    return oof, predictions


def run_kfold_nn(cfg, train, test, folds, num_features, cat_features, target, device, logger, n_fold=5, seed=7):

    oof = np.zeros((len(train), len(cfg.target_cols)))
    predictions = np.zeros((len(test), len(cfg.target_cols)))

    for _fold in range(n_fold):
        logger.info("Fold {}".format(_fold))
        _oof, _predictions = run_single_nn(cfg,
                                           train,
                                           test,
                                           folds,
                                           num_features,
                                           cat_features,
                                           target,
                                           device,
                                           logger,
                                           fold_num=_fold,
                                           seed=seed)
        oof += _oof
        predictions += _predictions / n_fold

    score = 0
    for i in range(target.shape[1]):
        _score = log_loss(target[:, i], oof[:, i])
        score += _score / target.shape[1]
    logger.info(f"CV score: {score}")

    return oof, predictions
