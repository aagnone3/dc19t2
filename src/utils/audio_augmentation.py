import numpy as np
import matplotlib.pyplot as plt
from librosa.feature import melspectrogram
from librosa.display import specshow
from librosa.effects import time_stretch, pitch_shift, harmonic, percussive
from scipy.signal import remez, freqz, lfilter


def lpf(x, wc=16000, n_taps=256, tw_factor=1/16.0, fs=44100):
    tw = wc * tw_factor
    band_edges = [0, wc, wc + tw, fs * 0.5]
    return remez(n_taps, band_edges, [1, 0], fs=fs)


def amplitude_scale(x, coeff=1.0):
    return x * coeff


def hp_reweight(x, lam):
    """
    Re-weight the audio as a convex combination of its estimated harmonic and percussive parts.
    Assuming x = x_h + x_p, compute x_new = 2 * lam * x_h + 2 * (1 - lam) * x_p.
    Note: a factor of 2 is multiplied by each term to retain the sum of the coefficients as 2.

    :param x: audio samples
    :param lam: weight for the harmonic part in [0, 1]
    :return : x_new
    """
    assert lam >= 0 and lam <= 1, 'lam must be in [0, 1]'
    return 2 * lam * harmonic(x) + 2 * (1 - lam) * percussive(x)
