# -*- coding: utf-8 -*-
#########################################################################
# Initial software
# Copyright Nicolas Turpault, Romain Serizel, Justin Salamon, Ankit Parag Shah, 2019, v1.0
# This software is distributed under the terms of the License MIT
#########################################################################
import argparse
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np


from DataLoad import DataLoadDf
from DatasetDcase2019Task1 import DatasetDcase2019Task1
from evaluation_measures import audio_tagging_results, get_f_measure_by_class, compute_strong_metrics, get_predictions
from utils.utils import ManyHotEncoder, to_cuda_if_available, get_transforms
from utils.Logger import LOG
from utils.Scaler import Scaler
from models.CRNN import CRNN
import config as cfg

from baseline.DataLoad import DataLoadDf as B_DataLoadDf
from baseline.DatasetDcase2019Task4 import DatasetDcase2019Task4 as B_DatasetDcase2019Task4


def test_model(state, reduced_number_of_data, strore_predicitions_fname=None):
    crnn_kwargs = state["model"]["kwargs"]
    crnn = CRNN(**crnn_kwargs)
    crnn.load(parameters=state["model"]["state_dict"])
    LOG.info("Model loaded at epoch: {}".format(state["epoch"]))
    pooling_time_ratio = state["pooling_time_ratio"]

    crnn.load(parameters=state["model"]["state_dict"])
    scaler = Scaler()
    scaler.load_state_dict(state["scaler"])
    classes = cfg.classes
    many_hot_encoder = ManyHotEncoder.load_state_dict(state["many_hot_encoder"])

    # ##############
    # Validation
    # ##############
    crnn = crnn.eval()
    [crnn] = to_cuda_if_available([crnn])
    transforms_valid = get_transforms(cfg.max_frames, scaler=scaler)

    # # 2018
    # LOG.info("Eval 2018")
    # eval_2018_df = dataset.initialize_and_get_df(cfg.eval2018, reduced_number_of_data)
    # # Strong
    # eval_2018_strong = DataLoadDf(eval_2018_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
    #                               transform=transforms_valid)
    # predictions = get_predictions(crnn, eval_2018_strong, many_hot_encoder.decode_strong)
    # compute_strong_metrics(predictions, eval_2018_df, pooling_time_ratio)
    # # Weak
    # eval_2018_weak = DataLoadDf(eval_2018_df, dataset.get_feature_file, many_hot_encoder.encode_weak,
    #                             transform=transforms_valid)
    # weak_metric = get_f_measure_by_class(crnn, len(classes), DataLoader(eval_2018_weak, batch_size=cfg.batch_size))
    # LOG.info("Weak F1-score per class: \n {}".format(pd.DataFrame(weak_metric * 100, many_hot_encoder.labels)))
    # LOG.info("Weak F1-score macro averaged: {}".format(np.mean(weak_metric)))

    # Validation 2019
    # LOG.info("Validation 2019 (original code)")
    # b_dataset = B_DatasetDcase2019Task4(cfg.workspace,
    #                                   base_feature_dir=os.path.join(cfg.workspace, 'dataset', 'features'),
    #                                   save_log_feature=False)
    # b_validation_df = b_dataset.initialize_and_get_df(cfg.validation, reduced_number_of_data)
    # b_validation_df.to_csv('old.csv')
    # b_validation_strong = B_DataLoadDf(b_validation_df,
    #                                  b_dataset.get_feature_file, many_hot_encoder.encode_strong_df,
    #                                  transform=transforms_valid)

    # predictions2 = get_predictions(crnn, b_validation_strong, many_hot_encoder.decode_strong,
    #                               save_predictions=strore_predicitions_fname)
    # compute_strong_metrics(predictions2, b_validation_df, pooling_time_ratio)

    # b_validation_weak = B_DataLoadDf(b_validation_df, b_dataset.get_feature_file, many_hot_encoder.encode_weak,
    #                              transform=transforms_valid)
    # weak_metric = get_f_measure_by_class(crnn, len(classes), DataLoader(b_validation_weak, batch_size=cfg.batch_size))
    # LOG.info("Weak F1-score per class: \n {}".format(pd.DataFrame(weak_metric * 100, many_hot_encoder.labels)))
    # LOG.info("Weak F1-score macro averaged: {}".format(np.mean(weak_metric)))

    # ============================================================================================
    # ============================================================================================
    # ============================================================================================

    dataset = DatasetDcase2019Task4(feature_dir=cfg.feature_dir,
                                    local_path=cfg.workspace,
                                    exp_tag=cfg.exp_tag,
                                    save_log_feature=False)
    # Validation 2019
    LOG.info("Validation 2019")
    validation_df = dataset.initialize_and_get_df(cfg.validation, reduced_number_of_data)
    validation_strong = DataLoadDf(validation_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                   transform=transforms_valid)

    predictions = get_predictions(crnn, validation_strong, many_hot_encoder.decode_strong,
                                  save_predictions=strore_predicitions_fname)
    vdf = validation_df.copy()
    vdf.filename = vdf.filename.str.replace('.npy', '.wav')
    pdf = predictions.copy()
    pdf.filename = pdf.filename.str.replace('.npy', '.wav')
    compute_strong_metrics(pdf, vdf, pooling_time_ratio)

    validation_weak = DataLoadDf(validation_df, dataset.get_feature_file, many_hot_encoder.encode_weak,
                                 transform=transforms_valid)
    weak_metric = get_f_measure_by_class(crnn, len(classes), DataLoader(validation_weak, batch_size=cfg.batch_size))
    LOG.info("Weak F1-score per class: \n {}".format(pd.DataFrame(weak_metric * 100, many_hot_encoder.labels)))
    LOG.info("Weak F1-score macro averaged: {}".format(np.mean(weak_metric)))

    # Just an example of how to get the weak predictions from dataframes.
    # print(audio_tagging_results(validation_df, predictions))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-s", '--subpart_data', type=int, default=None, dest="subpart_data",
                        help="Number of files to be used. Useful when testing on small number of files.")
    parser.add_argument("-m", '--model_path', type=str, default=None, dest="model_path",
                        help="Path of the model to be resume or to get validation results from.")
    parser.add_argument("-p", '--save_predictions_fname', type=str, default=None, dest="save_predictions_fname",
                        help="Path for the predictions to be saved, if not set, not save them")

    f_args = parser.parse_args()
    reduced_number_of_data = f_args.subpart_data
    model_path = f_args.model_path
    expe_state = torch.load(model_path, map_location="cpu")

    test_model(expe_state, reduced_number_of_data, f_args.save_predictions_fname)
