import soundfile as sf
import pyaudio
import pickle as pk
import numpy as np
import resampy
import pretty_midi
import json
import h5py
from tqdm import tqdm
from IPython.display import display, Audio
import librosa
import os
import itertools
import random
import librosa.display
import scipy.signal
import matplotlib.pyplot as plt
import noisereduce as nr
import sklearn
from audiolazy.lazy_synth import adsr
from dppy.finite_dpps import FiniteDPP
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from scipy.signal import hilbert
from collections.abc import Mapping

class LazyDict(Mapping):
    def __init__(self, keys, func):
        self._func = func
        self._keys = set(keys)

    def __getitem__(self, key):
        if key not in self._keys:
            raise KeyError(key)
        return self._func(key)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)


# Compute onset strength envelope
def qtile(a, q=45, *args, **kwargs):
    return np.percentile(a, q, *args, **kwargs)

