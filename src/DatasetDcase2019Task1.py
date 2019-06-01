# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import os
import librosa
import time
import pandas as pd
from tqdm import tqdm
from functools import partial

import config as cfg
from utils.Logger import LOG
from download_data import download
from utils.utils import create_folder, read_audio
from utils.audio_augmentation import *


class DatasetDcase2019Task1:
    """
    Args:
        local_path: str, (Default value = "") base directory where the dataset is, to be changed if
            dataset moved
        base_feature_dir: str, (Default value = "features) base directory to store the features
        recompute_features: bool, (Default value = False) wether or not to recompute features
        subpart_data: int, (Default value = None) allow to take only a small part of the dataset.
            This number represents the number of data to download and use from each set
        save_log_feature: bool, (Default value = True) whether or not saving the logarithm of the feature or not
            (particularly useful to put False to apply some data augmentation)

    Attributes:
        local_path: str, base directory where the dataset is, to be changed if
            dataset moved
        base_feature_dir: str, base directory to store the features
        recompute_features: bool, wether or not to recompute features
        subpart_data: int, allow to take only a small part of the dataset.
            This number represents the number of data to download and use from each set
        save_log_feature: bool, whether or not saving the logarithm of the feature or not
            (particularly useful to put False to apply some data augmentation)
        feature_dir : str, directory to store the features

    """
    def __init__(self, feature_dir, local_path="", recompute_features=False,
                 exp_tag='default', save_log_feature=True):

        self.local_path = local_path
        self.recompute_features = recompute_features
        self.save_log_feature = save_log_feature

        self.feature_dir = feature_dir
        # if feature_dir is None:
        #     feature_dir = os.path.join(base_feature_dir, '_' + exp_tag)
        # feature_dir = os.path.join(base_feature_dir, "sr" + str(cfg.sample_rate) + "_win" + str(cfg.n_window)
        #                            + "_hop" + str(cfg.hop_length) + "_mels" + str(cfg.n_mels))
        # if not self.save_log_feature:
        #     feature_dir += "_nolog"
        # self.feature_dir = os.path.join(feature_dir, "features")

        # create folder if not exist
        create_folder(self.feature_dir)

    def initialize_and_get_df(self, csv_path, subpart_data=None, download=True, training=False):
        """ Initialize the dataset, extract the features dataframes
        Args:
            csv_path: str, csv path in the initial dataset
            subpart_data: int, the number of file to take in the dataframe if taking a small part of the dataset.
            download: bool, whether or not to download the data from the internet (youtube).

        Returns:
            pd.DataFrame
            The dataframe containing the right features and labels
        """
        meta_name = os.path.join(self.local_path, csv_path)
        if download:
            self.download_from_meta(meta_name, subpart_data)
        return self.extract_features_from_meta(meta_name, subpart_data, training=training)

    @staticmethod
    def get_classes(list_dfs):
        """ Get the different classes of the dataset
        Returns:
            A list containing the classes
        """
        raise
        classes = []
        for df in list_dfs:
            classes.extend(df["label"].dropna().unique())
        return list(set(classes))

    @staticmethod
    def get_subpart_data(df, subpart_data):
        column = "filename"
        if not subpart_data > len(df[column].unique()):
            filenames = df[column].drop_duplicates().sample(subpart_data, random_state=10)
            df = df[df[column].isin(filenames)].reset_index(drop=True)
            LOG.debug("Taking subpart of the data, len : {}, df_len: {}".format(subpart_data, len(df)))
        return df

    @staticmethod
    def get_df_from_meta(meta_name, subpart_data=None):
        """
        Extract a pandas dataframe from a csv file

        Args:
            meta_name : str, path of the csv file to extract the df
            subpart_data: int, the number of file to take in the dataframe if taking a small part of the dataset.

        Returns:
            dataframe
        """
        df = pd.read_csv(meta_name, header=0, sep=",")
        if subpart_data is not None:
            df = DatasetDcase2019Task1.get_subpart_data(df, subpart_data)
        return df

    @staticmethod
    def get_audio_dir_path_from_meta(filepath):
        """ Get the corresponding audio dir from a meta filepath

        Args:
            filepath : str, path of the meta filename (csv)

        Returns:
            str
            path of the audio directory.
        """
        base_filepath = os.path.splitext(filepath)[0]
        audio_dir = os.path.dirname(base_filepath.replace("metadata", "audio"))
        audio_dir = '/'.join(audio_dir.split('/')[:-1])
        audio_dir = os.path.abspath(audio_dir)
        return audio_dir

    def download_from_meta(self, filename, subpart_data=None, n_jobs=3, chunk_size=10):
        """
        Download files contained in a meta file (csv)

        Args:
            filename: str, path of the meta file containing the name of audio files to donwnload
                (csv with column "filename")
            subpart_data: int, the number of files to use, if a subpart of the dataframe wanted.
            chunk_size: int, (Default value = 10) number of files to download in a chunk
            n_jobs : int, (Default value = 3) number of parallel jobs
        """
        result_audio_directory = self.get_audio_dir_path_from_meta(filename)
        # read metadata file and get only one filename once
        df = DatasetDcase2019Task1.get_df_from_meta(filename, subpart_data)
        filenames = df.filename.drop_duplicates()
        download(filenames, result_audio_directory, n_jobs=n_jobs, chunk_size=chunk_size)

    def get_feature_file(self, filename):
        """
        Get a feature file from a filename
        Args:
            filename:  str, name of the file to get the feature

        Returns:
            numpy.array
            containing the features computed previously
        """
        fname = os.path.join(self.feature_dir, os.path.splitext(filename)[0] + ".npy")
        data = np.load(fname)
        if np.any(np.isnan(data)):
            raise ValueError('nan features in {}'.format(filename))
        return data

    def calculate_mel_spec(self, audio):
        """
        Calculate a mal spectrogram from raw audio waveform
        Note: The parameters of the spectrograms are in the config.py file.
        Args:
            audio : numpy.array, raw waveform to compute the spectrogram

        Returns:
            numpy.array
            containing the mel spectrogram
        """
        # Compute spectrogram
        ham_win = np.hamming(cfg.n_window)

        spec = librosa.stft(
            audio,
            n_fft=cfg.n_window,
            hop_length=cfg.hop_length,
            window=ham_win,
            center=True,
            pad_mode='reflect'
        )

        mel_spec = librosa.feature.melspectrogram(
            S=np.abs(spec),  # amplitude, for energy: spec**2 but don't forget to change amplitude_to_db.
            sr=cfg.sample_rate,
            n_mels=cfg.n_mels,
            fmin=cfg.f_min, fmax=cfg.f_max,
            htk=False, norm=None)

        if self.save_log_feature:
            mel_spec = librosa.amplitude_to_db(mel_spec)  # 10 * log10(S**2 / ref), ref default is 1
        mel_spec = mel_spec.T
        mel_spec = mel_spec.astype(np.float32)
        return mel_spec

    def extract_features_from_meta(self, csv_audio, subpart_data=None, training=False):
        """Extract log mel spectrogram features.

        Args:
            csv_audio : str, file containing names, durations and labels : (name, start, end, label, label_index)
                the associated wav_filename is Yname_start_end.wav
            subpart_data: int, number of files to extract features from the csv.
        """
        t1 = time.time()
        df_meta = self.get_df_from_meta(csv_audio, subpart_data)
        df_all = list()
        feature_fns = list()
        LOG.info('Extracting/loading features')
        LOG.info("{} Total file number: {}".format(csv_audio, len(df_meta.filename.unique())))

        augmentation_funcs = [
            ('orig', None),  # original signal
        ]

        if training:
            augmentation_funcs += [
                # ('lpf4k', partial(lpf, wc=4000, fs=cfg.sample_rate)),
                # ('lpf8k', partial(lpf, wc=8000, fs=cfg.sample_rate)),
                # ('lpf16k', partial(lpf, wc=16000, fs=cfg.sample_rate)),
                # ('ps-6', partial(pitch_shift, sr=cfg.sample_rate, n_steps=-6)),
                # ('ps-3', partial(pitch_shift, sr=cfg.sample_rate, n_steps=-3)),
                # ('ps+3', partial(pitch_shift, sr=cfg.sample_rate, n_steps=3)),
                # ('ps+6', partial(pitch_shift, sr=cfg.sample_rate, n_steps=6)),
                # ('ts1.25', partial(time_stretch, rate=1.25)),
                # ('ts1.5', partial(time_stretch, rate=1.5)),
                # ('amp0.5', partial(amplitude_scale, coeff=0.5)),
                # ('amp0.75', partial(amplitude_scale, coeff=0.75)),
                # ('hp0.25', partial(hp_reweight, lam=0.25)),
                # ('hp0.75', partial(hp_reweight, lam=0.75))
            ]

        wav_fns = df_meta.filename.unique()
        flag = False
        for ind, wav_name in tqdm(enumerate(wav_fns), total=len(wav_fns)):
            if ind % 500 == 0:
                LOG.debug(ind)

            # verify the audio file is present
            wav_dir = self.get_audio_dir_path_from_meta(csv_audio)
            wav_path = os.path.join(wav_dir, wav_name)
            if os.path.isfile(wav_path):
                # defer loading audio until the need for feature extraction is verified
                audio = None

                # perform all augmentations (including no augmentation)
                for name, func in augmentation_funcs:
                    if name == 'orig':
                        out_filename = os.path.splitext(wav_name)[0] + ".npy"
                    else:
                        out_filename = os.path.splitext(wav_name)[0] + '_' + name + ".npy"
                    out_path = os.path.join(self.feature_dir, out_filename)

                    # add the metadata
                    meta = df_meta.loc[df_meta.filename == wav_name]
                    df_all.append(meta)

                    # for synthetic data with time annotation of events, the meta df will have several entries for
                    # each wav file. therefore, we need to append the feature filename len(meta) times.
                    if len(meta) > 1:
                        feature_fns += [out_filename] * len(meta)
                        if flag:
                            print('Length of meta: {}'.format(len(meta)))
                            flag = False
                    else:
                        feature_fns.append(out_filename)

                    if not os.path.exists(out_path):
                        if audio is None:
                            (audio, _) = read_audio(wav_path, cfg.sample_rate)
                            if audio.shape[0] == 0:
                                print("File %s is corrupted!" % wav_path)
                                del feature_fns[-1]
                                del df_all[-1]

                        # perform any augmentation, extract features, save features
                        # LOG.info('extracting {}'.format(out_filename))
                        if func is not None:
                            mel_spec = self.calculate_mel_spec(func(audio))
                        else:
                            mel_spec = self.calculate_mel_spec(audio)
                        np.save(out_path, mel_spec)

                        LOG.debug("compute features time: %s" % (time.time() - t1))
            else:
                LOG.error("File %s is in the csv file but the feature is not extracted!" % wav_path)
                # df_meta = df_meta.drop(df_meta[df_meta.filename == wav_name].index)

        # form the final DataFrame of meta data for features from original and augmented audio
        df_all = pd.concat(df_all).reset_index(drop=True)
        df_all['feature_filename'] = feature_fns

        return df_all
