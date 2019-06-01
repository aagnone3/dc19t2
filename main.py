import os
import yaml
import argparse
import pickle
from os import path
import pandas as pd
import numpy as np
from sklearn.externals import joblib

from utils import factory, create_dirs
from evaluation import prfs_multilabel
from evaluation.dcase2019_task4 import DCASE2019T4Evaluator

np.random.seed(44)


def process_config(fn):
    with open(fn, 'r') as fp:
        config = yaml.load(fp)

    config["callbacks"]["tensorboard_log_dir"] = os.path.join(
        "experiments",
        config["exp"]["name"],
        "logs/"
    )
    config["callbacks"]["checkpoint_dir"] = os.path.join(
        "experiments",
        config["exp"]["name"],
        "checkpoints/"
    )
    return config


def get_data(config, extract_real_time=False):
    # define directory names
    train_data_dir = config["data_loader"]["train"]["data_dir"]
    eval_data_dir = config["data_loader"]["eval"]["data_dir"]

    # load data set meta data and labels
    all_train_data_df = pd.read_csv(config["data_loader"]["train"]["meta_file"]).fillna('nan')

    train_idx = np.random.choice(all_train_data_df.index, size=int(0.7 * len(all_train_data_df)))
    train_mask = all_train_data_df.index.isin(train_idx)
    train_df, training_val_df = all_train_data_df.loc[train_mask], all_train_data_df.loc[~train_mask]

    eval_df = pd.read_csv(config["data_loader"]["eval"]["meta_file"]).fillna('nan')

    if len(train_df) == 0:
        raise ValueError("train_df is empty")
    if len(training_val_df) == 0:
        raise ValueError("training_val_df is empty")
    if len(eval_df) == 0:
        raise ValueError("eval_df is empty")

    if extract_real_time:
        train_df['filename'] = train_df['filename'].map(lambda fn: path.join(train_data_dir, fn))
        training_val_df['filename'] = training_val_df['filename'].map(lambda fn: path.join(train_data_dir, fn))
        eval_df['fname'] = eval_df['filename'].map(lambda fn: path.join(eval_data_dir, fn))
    else:
        data_dir = "features"
        train_df['filename'] = train_df['filename'].map(lambda fn: path.join(data_dir, fn) + '.hdf5')
        training_val_df['filename'] = training_val_df['filename'].map(lambda fn: path.join(data_dir, fn) + '.hdf5')
        eval_df['filename'] = eval_df['filename'].map(lambda fn: path.join(data_dir, fn) + '.hdf5')

    return train_df, training_val_df, eval_df


def show_metrics(model, evaluation):
    for metric_name, metric_value in zip(model.model.metrics_names, evaluation):
        print("{}: {}".format(metric_name, metric_value))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    parser.add_argument(
        '-m', '--mode',
        dest='mode',
        metavar='M',
        choices=['train', 'eval', 'predict'],
        default='train',
        required=False,
        help='Mode to run in.')
    parser.add_argument(
        '-T', '--tri-training',
        dest='tri_training',
        required=False,
        default=False,
        action='store_true',
        help='Flag to use the tri-training SSL technique.')
    return parser.parse_args()


def main():
    args = parse_args()
    config = process_config(args.config)

    # create the experiments dirs
    create_dirs([config["callbacks"]["tensorboard_log_dir"], config["callbacks"]["checkpoint_dir"]])

    # resolve classes
    data_loader_cls = factory("data_loaders.{}".format(config["data_loader"]["name"]))
    trainer_cls = factory("trainers.{}".format(config["trainer"]["name"]))

    # create data generators for the data sets
    loader_params = config["data_loader"]
    generators = {
        name: data_loader_cls(name, **loader_params)
        for name in ['train', 'eval', 'unlabeled']
    }

    # train ze model(s)
    if args.tri_training:
        model = list()
        for i in range(1, 4):
            print('Creating model {}/3'.format(i))
            model_params = config['model{}'.format(i)]
            model_cls = factory("models.{}".format(config["model{}".format(i)]["name"]))
            model.append(model_cls(generators['train'].feature_dim, generators['train'].n_classes, **model_params))
    else:
        model_params = config['model']
        model_cls = factory("models.{}".format(config["model"]["name"]))
        model = model_cls(generators['train'].feature_dim, generators['train'].n_classes, **model_params)

    trainer = trainer_cls(
        config["exp"]["name"],
        model,
        config["callbacks"],
        **config["trainer"]
    )
    if args.mode == 'train':
        trainer.train(
            generators['train'],
            generators['eval'],
            # generators['unlabeled']
            # confidence_threshold=0
        )
    elif args.mode == 'eval':
        evaluator = DCASE2019T4Evaluator(
            trainer.predict,
            generators['eval'],
            verbose=True
        )
        ret = evaluator.evaluate(evaluator.find_class_thresholds())
    else:
        # mode == 'predict'
        for data_type, fns in config['trainer']['prediction'].items():
            print('Predicting on {} data and storing at {}.'.format(data_type, fns['predictions_fn']))
            trainer.predict(
                generators[data_type],
                write_fn=path.join('experiments', config['exp']['name'], fns['predictions_fn'])
            )


if __name__ == '__main__':
    main()
