# -*- coding: utf-8 -*-
#########################################################################
# This file is derived from Curious AI/mean-teacher, under the Creative Commons Attribution-NonCommercial
# Copyright Nicolas Turpault, Romain Serizel, Justin Salamon, Ankit Parag Shah, 2019, v1.0
# This software is distributed under the terms of the License MIT
#########################################################################

import argparse
import os
import time
import shutil
from os import path

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn

from utils import ramps
from DatasetDcase2019Task1 import DatasetDcase2019Task1
from DataLoad import DataLoadDf, ConcatDataset, MultiStreamBatchSampler
from utils.Scaler import Scaler
from TestModel import test_model
from evaluation_measures import get_f_measure_by_class, get_predictions, audio_tagging_results, compute_strong_metrics
from models.CRNN import CRNN
import config as cfg
from utils.utils import ManyHotEncoder, create_folder, SaveBest, to_cuda_if_available, weights_init, \
    get_transforms, AverageMeterSet, LDSLoss
from utils.Logger import LOG
torch.manual_seed(0)


def adjust_learning_rate(optimizer, rampup_value, rampdown_value):
    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = rampup_value * rampdown_value * cfg.max_learning_rate
    beta1 = rampdown_value * cfg.beta1_before_rampdown + (1. - rampdown_value) * cfg.beta1_after_rampdown
    beta2 = (1. - rampup_value) * cfg.beta2_during_rampdup + rampup_value * cfg.beta2_after_rampup
    weight_decay = (1 - rampup_value) * cfg.weight_decay_during_rampup + cfg.weight_decay_after_rampup * rampup_value

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['betas'] = (beta1, beta2)
        param_group['weight_decay'] = weight_decay


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(cfg, train_loader, model, optimizer, epoch, ema_model=None, weak_mask=None, strong_mask=None):
    """ One epoch of a Mean Teacher model
    :param train_loader: torch.utils.data.DataLoader, iterator of training batches for an epoch.
    Should return 3 values: teacher input, student input, labels
    :param model: torch.Module, model to be trained, should return a weak and strong prediction
    :param optimizer: torch.Module, optimizer used to train the model
    :param epoch: int, the current epoch of training
    :param ema_model: torch.Module, student model, should return a weak and strong prediction
    :param weak_mask: mask the batch to get only the weak labeled data (used to calculate the loss)
    :param strong_mask: mask the batch to get only the strong labeled data (used to calcultate the loss)
    """
    class_criterion = nn.BCELoss()
    consistency_criterion_strong = nn.MSELoss()
    lds_criterion = LDSLoss(xi=cfg.vat_xi, eps=cfg.vat_eps, n_power_iter=cfg.vat_n_power_iter)
    [class_criterion, consistency_criterion_strong, lds_criterion] = to_cuda_if_available([
        class_criterion,
        consistency_criterion_strong,
        lds_criterion
    ])

    meters = AverageMeterSet()

    LOG.debug("Nb batches: {}".format(len(train_loader)))
    start = time.time()
    rampup_length = len(train_loader) * cfg.n_epoch // 2
    for i, (batch_input, ema_batch_input, target) in enumerate(train_loader):
        global_step = epoch * len(train_loader) + i
        if global_step < rampup_length:
            rampup_value = ramps.sigmoid_rampup(global_step, rampup_length)
        else:
            rampup_value = 1.0

        # Todo check if this improves the performance
        # adjust_learning_rate(optimizer, rampup_value, rampdown_value)
        meters.update('lr', optimizer.param_groups[0]['lr'])

        [batch_input, ema_batch_input, target] = to_cuda_if_available([batch_input, ema_batch_input, target])
        LOG.debug(batch_input.mean())
        # Outputs
        strong_pred_ema, weak_pred_ema = ema_model(ema_batch_input)
        strong_pred_ema = strong_pred_ema.detach()
        weak_pred_ema = weak_pred_ema.detach()

        strong_pred, weak_pred = model(batch_input)
        loss = None
        # Weak BCE Loss
        # Take the max in axis 2 (assumed to be time)
        if len(target.shape) > 2:
            target_weak = target.max(-2)[0]
        else:
            target_weak = target

        if weak_mask is not None:
            weak_class_loss = class_criterion(weak_pred[weak_mask], target_weak[weak_mask])
            ema_class_loss = class_criterion(weak_pred_ema[weak_mask], target_weak[weak_mask])

            if i == 0:
                LOG.debug("target: {}".format(target.mean(-2)))
                LOG.debug("Target_weak: {}".format(target_weak))
                LOG.debug("Target_weak mask: {}".format(target_weak[weak_mask]))
                LOG.debug(weak_class_loss)
                LOG.debug("rampup_value: {}".format(rampup_value))
            meters.update('weak_class_loss', weak_class_loss.item())

            meters.update('Weak EMA loss', ema_class_loss.item())

            loss = weak_class_loss

        # Strong BCE loss
        if strong_mask is not None:
            strong_class_loss = class_criterion(strong_pred[strong_mask], target[strong_mask])
            meters.update('Strong loss', strong_class_loss.item())

            strong_ema_class_loss = class_criterion(strong_pred_ema[strong_mask], target[strong_mask])
            meters.update('Strong EMA loss', strong_ema_class_loss.item())
            if loss is not None:
                loss += strong_class_loss
            else:
                loss = strong_class_loss

        # Teacher-student consistency cost
        if ema_model is not None:

            consistency_cost = cfg.max_consistency_cost * rampup_value
            meters.update('Consistency weight', consistency_cost)
            # Take only the consistence with weak and unlabel
            consistency_loss_strong = consistency_cost * consistency_criterion_strong(strong_pred,
                                                                                      strong_pred_ema)
            meters.update('Consistency strong', consistency_loss_strong.item())
            if loss is not None:
                loss += consistency_loss_strong
            else:
                loss = consistency_loss_strong

            meters.update('Consistency weight', consistency_cost)
            # Take only the consistence with weak and unlabel
            consistency_loss_weak = consistency_cost * consistency_criterion_strong(weak_pred, weak_pred_ema)
            meters.update('Consistency weak', consistency_loss_weak.item())
            if loss is not None:
                loss += consistency_loss_weak
            else:
                loss = consistency_loss_weak

        # LDS loss
        if cfg.vat_enabled:
            lds_loss = cfg.vat_coeff * lds_criterion(model, batch_input, weak_pred)
            LOG.info('loss: {:.3f}, lds loss: {:.3f}'.format(loss, cfg.vat_coeff * lds_loss.detach().cpu().numpy()))
            loss += lds_loss
        else:
            if i % 25 == 0:
                LOG.info('loss: {:.3f}'.format(loss))

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        assert not loss.item() < 0, 'Loss problem, cannot be negative'
        meters.update('Loss', loss.item())

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        if ema_model is not None:
            update_ema_variables(model, ema_model, 0.999, global_step)

    epoch_time = time.time() - start

    LOG.info(
        'Epoch: {}\t'
        'Time {:.2f}\t'
        '{meters}'.format(
            epoch, epoch_time, meters=meters))


if __name__ == '__main__':
    LOG.info("MEAN TEACHER")
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-s", '--subpart_data', type=int, default=None, dest="subpart_data",
                        help="Number of files to be used. Useful when testing on small number of files.")

    parser.add_argument("-n", '--no_synthetic', dest='no_synthetic', action='store_true', default=False,
                        help="Not using synthetic labels during training")
    f_args = parser.parse_args()

    reduced_number_of_data = f_args.subpart_data
    no_synthetic = f_args.no_synthetic
    LOG.info("subpart_data = {}".format(reduced_number_of_data))
    LOG.info("Using synthetic data = {}".format(not no_synthetic))

    store_dir = os.path.join("stored_data", cfg.exp_tag)
    saved_model_dir = os.path.join(store_dir, "model")
    saved_pred_dir = os.path.join(store_dir, "predictions")
    create_folder(store_dir)
    create_folder(saved_model_dir)
    create_folder(saved_pred_dir)
    shutil.copyfile('src/config.py', os.path.join(store_dir, 'config.py'))

    pooling_time_ratio = cfg.pooling_time_ratio  # --> Be careful, it depends of the model time axis pooling
    # ##############
    # DATA
    # ##############
    dataset = DatasetDcase2019Task1(feature_dir=cfg.feature_dir,
                                    local_path=cfg.workspace,
                                    exp_tag=cfg.exp_tag,
                                    save_log_feature=False)

    train_df = dataset.initialize_and_get_df(cfg.train, reduced_number_of_data, training=True)
    validation_df = dataset.initialize_and_get_df(cfg.validation, reduced_number_of_data, training=True)
    test_df = dataset.initialize_and_get_df(cfg.test, reduced_number_of_data, training=True)

    many_hot_encoder = ManyHotEncoder(cfg.classes, n_frames=cfg.max_frames // pooling_time_ratio)
    transforms = get_transforms(cfg.max_frames)

    train_data = DataLoadDf(train_df, dataset.get_feature_file, many_hot_encoder.encode_weak, transform=transforms)
    validation_data = DataLoadDf(validation_df, dataset.get_feature_file, many_hot_encoder.encode_weak,
                                 transform=transforms)
    test_data = DataLoadDf(test_df, dataset.get_feature_file, many_hot_encoder.encode_weak, transform=transforms)

    list_dataset = [train_data]
    batch_sizes = [cfg.batch_size]
    # batch_sizes = [cfg.batch_size // len(list_dataset)] * len(list_dataset)
    weak_mask = slice(cfg.batch_size)
    strong_mask = None

    scaler = Scaler()
    if path.exists(cfg.scaler_fn):
        LOG.info('Loading scaler from {}'.format(cfg.scaler_fn))
        scaler.load(cfg.scaler_fn)
    else:
        scaler.calculate_scaler(ConcatDataset(list_dataset))
        LOG.info('Saving scaler to {}'.format(cfg.scaler_fn))
        scaler.save(cfg.scaler_fn)

    LOG.debug(scaler.mean_)

    transforms = get_transforms(cfg.max_frames, scaler, augment_type="noise")
    transforms_valid = get_transforms(cfg.max_frames, scaler=scaler)
    for i in range(len(list_dataset)):
        list_dataset[i].set_transform(transforms)
    validation_data.set_transform(transforms_valid)
    test_data.set_transform(transforms_valid)

    concat_dataset = ConcatDataset(list_dataset)
    sampler = MultiStreamBatchSampler(concat_dataset,
                                      batch_sizes=batch_sizes)
    training_data = DataLoader(concat_dataset, batch_sampler=sampler)

    # ##############
    # Model
    # ##############
    crnn_kwargs = cfg.crnn_kwargs
    crnn = CRNN(**crnn_kwargs)
    crnn_ema = CRNN(**crnn_kwargs)

    if path.exists(cfg.load_weights_fn):
        model_cfg = torch.load(cfg.load_weights_fn)
        crnn.load(parameters=model_cfg['model']['state_dict'])
        update_ema_variables(crnn, crnn_ema, 0.999, 0)
    else:
        crnn.apply(weights_init)
        crnn_ema.apply(weights_init)
    LOG.info(crnn)

    for param in crnn_ema.parameters():
        param.detach_()

    optim_kwargs = {"lr": 0.001, "betas": (0.9, 0.999)}
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, crnn.parameters()), **optim_kwargs)
    bce_loss = nn.BCELoss()

    state = {
        'model': {"name": crnn.__class__.__name__,
                  'args': '',
                  "kwargs": crnn_kwargs,
                  'state_dict': crnn.state_dict()},
        'model_ema': {"name": crnn_ema.__class__.__name__,
                      'args': '',
                      "kwargs": crnn_kwargs,
                      'state_dict': crnn_ema.state_dict()},
        'optimizer': {"name": optimizer.__class__.__name__,
                      'args': '',
                      "kwargs": optim_kwargs,
                      'state_dict': optimizer.state_dict()},
        "pooling_time_ratio": pooling_time_ratio,
        "scaler": scaler.state_dict(),
        "many_hot_encoder": many_hot_encoder.state_dict()
    }

    save_best_cb = SaveBest("sup")

    # ##############
    # Train
    # ##############
    for epoch in range(cfg.n_epoch):
        crnn = crnn.train()
        crnn_ema = crnn_ema.train()

        [crnn, crnn_ema] = to_cuda_if_available([crnn, crnn_ema])

        train(cfg, training_data, crnn, optimizer, epoch, ema_model=crnn_ema, weak_mask=weak_mask, strong_mask=strong_mask)

        crnn = crnn.eval()
        LOG.info("\n ### Validation Metrics ### \n")
        # predictions = get_predictions(crnn, validation_data, many_hot_encoder.decode_strong,
        #                               save_predictions=None)
        # pdf = predictions.copy()
        # pdf.filename = pdf.filename.str.replace('.npy', '.wav')
        # valid_events_metric = compute_strong_metrics(pdf, valid_synth_df, pooling_time_ratio)

        # LOG.info("\n ### Valid weak metric ### \n")
        weak_metric = get_f_measure_by_class(crnn, len(cfg.classes),
                                             DataLoader(validation_data, batch_size=cfg.batch_size))

        LOG.info("Weak F1-score per class: \n {}".format(pd.DataFrame(weak_metric * 100, many_hot_encoder.labels)))
        LOG.info("Weak F1-score macro averaged: {}".format(np.mean(weak_metric)))

        state['model']['state_dict'] = crnn.state_dict()
        state['model_ema']['state_dict'] = crnn_ema.state_dict()
        state['optimizer']['state_dict'] = optimizer.state_dict()
        state['epoch'] = epoch
        # state['valid_metric'] = valid_events_metric.results()
        if cfg.checkpoint_epochs is not None and (epoch + 1) % cfg.checkpoint_epochs == 0:
            model_fname = os.path.join(saved_model_dir, '_epoch_' + str(epoch))
            torch.save(state, model_fname)

        if cfg.save_best:
            global_valid = np.mean(weak_metric)
            if save_best_cb.apply(global_valid):
                model_fname = os.path.join(saved_model_dir, '_best')
                torch.save(state, model_fname)

    if cfg.save_best:
        model_fname = os.path.join(saved_model_dir, '_best')
        state = torch.load(model_fname)
        LOG.info("testing model: {}".format(model_fname))
    else:
        LOG.info("testing model of last epoch: {}".format(cfg.n_epoch))

    # ##############
    # Validation
    # ##############
    predictions_fname = os.path.join(saved_pred_dir, "_validation.csv")
    test_model(state, reduced_number_of_data, predictions_fname)
