import librosa
import numpy as np
from typing import List
from ambientmidi.features import get_feature_dict
from ambientmidi.utils import qtile

# Get onsets
DEFAULT_SPEC_KWARGS = dict(
    sample_rate = 16000,
    window_size_ms = 25,
    hop_size_ms = 10,
    min_clip_size_s = 0.125,
    max_clip_size_s = 1.0,
    n_fft = int((25 / 1000) * 16000), # 25 ms @ 16kHz
    hop_length = int((10 / 1000) * 16000), # 10 ms @ 16kHz
    n_mels = 40,
)

def update_kwargs(**kwargs):
    assert set(kwargs.keys()).issubset(set(DEFAULT_SPEC_KWARGS.keys()))
    sample_rate = kwargs.get("sample_rate", DEFAULT_SPEC_KWARGS["sample_rate"])
    window_size_ms = kwargs.get("window_size_ms", DEFAULT_SPEC_KWARGS["window_size_ms"])
    hop_size_ms = kwargs.get("hop_size_ms", DEFAULT_SPEC_KWARGS["hop_size_ms"])

    out_kwargs = dict(**DEFAULT_SPEC_KWARGS)
    for k in DEFAULT_SPEC_KWARGS.keys():
        if k in kwargs:
            if k == "n_fft":
                out_kwargs[k] = int((window_size_ms / 1000) * sample_rate)
            elif k == "hop_length":
                out_kwargs[k] = int((hop_size_ms / 1000) * sample_rate)
            else:
                out_kwargs[k] = kwargs[k]
    return out_kwargs


# Compute PCEN-gram (on mel-scale)
def compute_pcengram(audio, **kwargs):
    kwargs = update_kwargs(**kwargs)
    sample_rate = kwargs["sample_rate"]
    n_fft = kwargs["n_fft"]
    hop_length = kwargs["hop_length"]
    n_mels = kwargs["n_mels"]
    stft = librosa.stft(
            audio,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length)
    melspec = librosa.feature.melspectrogram(
        S=(stft.real*stft.real) + (stft.imag*stft.imag),
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        htk=True,
    )
    return librosa.pcen(S=melspec, sr=sample_rate, hop_length=hop_length)


def get_onset_idxs(
    pcengram: np.ndarray,
    **spec_kwargs,
):
    sample_rate = spec_kwargs.get("sample_rate", DEFAULT_SPEC_KWARGS["sample_rate"])
    n_fft = spec_kwargs.get("n_fft", DEFAULT_SPEC_KWARGS["n_fft"])
    hop_length = spec_kwargs.get("hop_length", DEFAULT_SPEC_KWARGS["hop_lenth"])
    n_mels = spec_kwargs.get("n_mels", DEFAULT_SPEC_KWARGS["n_mels"])
    onset_env = librosa.onset.onset_strength(
        S=pcengram,
        sr=sample_rate,
        aggregate=qtile,
        n_mels=n_mels,
        hop_length=hop_length,
    )
    # Detect onsets from envelope
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sample_rate,
        hop_length=hop_length,
        units='frames',
        backtrack=True,
        delta=0.03,
        wait=0.03 * sample_rate // hop_length,
        pre_max=0.03 * sample_rate // hop_length,
        post_max=0.03 * sample_rate // hop_length,
        pre_avg=0.10 * sample_rate // hop_length,
        post_avg=0.10 * sample_rate // hop_length,
    )
    return librosa.core.frames_to_samples(onset_frames, hop_length=hop_length)

def truncate_silence(audio, n_fft, hop_length, min_clip_length):
    rms = librosa.feature.rms(
        audio, frame_length=n_fft, hop_length=hop_length,
    ).flatten()
    thresh = rms.max() * 0.01
    silent_frame_idxs = [
        idx for idx in np.nonzero(rms <= thresh)[0]
        if idx >= min_clip_length
    ]
    if len(silent_frame_idxs) > 0:
        end_idx = librosa.frames_to_samples(
            silent_frame_idxs[0] + 1,
            hop_length=hop_length,
            n_fft=n_fft,
        )
        audio = audio[:end_idx]

    return audio


def get_event_clip_dicts(audio: np.ndarray, onset_idx_list: List, truncate_silence=False, **spec_kwargs):
    sample_rate = spec_kwargs.get("sample_rate", DEFAULT_SPEC_KWARGS["sample_rate"])
    n_fft = spec_kwargs.get("n_fft", DEFAULT_SPEC_KWARGS["n_fft"])
    hop_length = spec_kwargs.get("hop_length", DEFAULT_SPEC_KWARGS["hop_length"])
    min_clip_length = spec_kwargs.get("min_clip_length", DEFAULT_SPEC_KWARGS["min_clip_length"]) * sample_rate
    max_clip_length = spec_kwargs.get("max_clip_length", DEFAULT_SPEC_KWARGS["max_clip_length"]) * sample_rate

    # Compute features for event clips
    clip_list = []
    for onset_idx in onset_idx_list:
        # Get clip audio
        end_idx = min(onset_idx + max_clip_length, audio.shape[0])
        clip = audio[onset_idx:end_idx]
        if truncate_silence:
            clip = truncate_silence(clip, n_fft, hop_length, min_clip_length)

        clip_dict = get_feature_dict(clip, sample_rate)
        clip_list.append(clip_dict)
    return clip_list