#import openl3
import numpy as np
import librosa
#import magenta
import resampy
import pretty_midi
from scipy.signal import hilbert
from constants import SAMPLE_RATE
import os

PROJECT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..'))
MFCC_NUM_COEFFS = 40

NSYNTH_CKPT_PATH = os.path.join(PROJECT_DIR,
                                "models",
                                "wavenet-ckpt",
                                "model.ckpt-200000")


def mfcc(audio, sr):
    mfcc_arr = librosa.feature.mfcc(y=audio,
                                    sr=sr,
                                    n_mfcc=MFCC_NUM_COEFFS+1)[1:]
    return mfcc_arr.mean(axis=-1)


def openl3(audio, sr):
    emb, ts = openl3.get_audio_embedding(audio, sr,
                                         content_type="music",
                                         input_repr="mel128",
                                         embedding_size=512)
    return emb.mean(axis=0)


def nsynth(audio, sr):
    from magenta.models.nsynth.wavenet import fastgen
    if sr != SAMPLE_RATE:
        audio = resampy.resample(audio, sr, SAMPLE_RATE)
    sample_length = len(audio)
    emb = fastgen.encode(audio, NSYNTH_CKPT_PATH, sample_length)
    emb = emb.squeeze(axis=0)
    return emb.mean(axis=0)


def envelope(audio):
    analytic = hilbert(audio)
    return np.abs(analytic)


def pitch(audio, sr):
    # Could also use crepe but we already have lots of deep learning
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y=audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr)
    return f0[voiced_flag].mean()


def tonality(audio):
    sf = librosa.feature.spectral_flatness(y=audio)
    return 1 - sf.mean()


def get_feature_dict(audio, sr, no_audio=False):
    #openl3_emb = openl3(audio, sr)
    if sr != SAMPLE_RATE:
        audio = resampy.resample(audio, sr, SAMPLE_RATE)
    sr = SAMPLE_RATE
    res = {
        'mfcc': mfcc(audio, sr),
        'pitch_hz': pitch(audio, sr),
        'tonality': tonality(audio),
        # 'openl3': openl3_emb,
        # 'nsynth': nsynth(audio, sr),
        # 'envelope': envelope(audio),
    }
    res['pitch'] = pretty_midi.hz_to_note_number(res['pitch_hz'])
    if not no_audio:
        res['audio'] = audio
    return res


