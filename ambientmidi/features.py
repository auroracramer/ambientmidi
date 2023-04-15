import numpy as np
import pretty_midi
import librosa
import os
import librosa.display
from scipy.signal import hilbert


FEATURE_FUNCTIONS = {}
def _feature(fn):
    feat_name = fn.__name__.split('_', 1)[1]
    FEATURE_FUNCTIONS[feat_name] = fn
    return fn


MFCC_NUM_COEFFS = 40
@_feature
def compute_mfcc(audio, sr):
    mfcc_arr = librosa.feature.mfcc(y=audio,
                                    sr=sr,
                                    n_mfcc=MFCC_NUM_COEFFS+1)[1:]
    return mfcc_arr.mean(axis=-1)


@_feature
def compute_pitch_hz(audio, sr):
    # Could also use crepe but we already have lots of deep learning
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y=audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr)
    return f0[voiced_flag].mean()

@_feature
def compute_pitch(audio, sr):
    pitch_hz = compute_pitch_hz(audio, sr)
    return pretty_midi.hz_to_note_number(pitch_hz)


@_feature
def compute_tonality(audio, sr):
    sf = librosa.feature.spectral_flatness(y=audio)
    return 1 - sf.mean()


@_feature
def compute_openl3(audio, sr):
    import openl3
    emb, _ = openl3.get_audio_embedding(
        audio, sr,
        content_type="music",
        input_repr="mel128",
        embedding_size=512
    )
    return emb.mean(axis=0)
                    

@_feature
def compute_nsynth(audio, sr):
    # probably need to fix
    PROJECT_DIR = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..'
        )
    )
    
    NSYNTH_CKPT_PATH = os.path.join(
        PROJECT_DIR,
        "models",
        "wavenet-ckpt",
        "model.ckpt-200000"
    ) 
    from magenta.models.nsynth.wavenet import fastgen
    sample_length = len(audio)
    emb = fastgen.encode(audio, NSYNTH_CKPT_PATH, sample_length)
    emb = emb.squeeze(axis=0)
    return emb.mean(axis=0)


@_feature
def compute_envelope(audio, sr):
    analytic = hilbert(audio)
    return np.abs(analytic)


@_feature
def compute_audio(audio, sr):
    return audio


def get_feature_dict(audio, sr, features=("audio", "mfcc", "pitch", "pitch_hz", "tonality")):
    res = {}
    for feat_name in features:
        if feat_name not in FEATURE_FUNCTIONS:
            raise ValueError(f"Invalid feature: {feat_name}")
        res[feat_name] = FEATURE_FUNCTIONS[feat_name](audio, sr)
    return res