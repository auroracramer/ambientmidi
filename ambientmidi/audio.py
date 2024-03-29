import pyaudio
import resampy
import soundfile as sf
import numpy as np
import noisereduce as nr
import pyloudnorm as ln

SAMPLE_RATE = 16000



def record_audio(duration: float = 60.0, sample_rate: int = 16000, denoise=True):
    # [duration] = seconds
    p = pyaudio.PyAudio()
    inp_stream = p.open(
        format=pyaudio.paFloat32, channels=1, rate=sample_rate, input=True,
    )
    audio = inp_stream.read(int(duration * sample_rate))
    audio = np.frombuffer(audio, dtype=np.float32)
    inp_stream.stop_stream()
    inp_stream.close()
    audio = rescale_audio(audio)
    if denoise:
        audio = nr.reduce_noise(y=audio, sr=sample_rate)
    return audio


def load_audio(path, sample_rate):
    audio, sr = sf.read(str(path), always_2d=True)
    audio = audio.mean(axis=-1) # convert to mono
    if sr != sample_rate:
        audio = resampy.resample(audio, sr, sample_rate)
    return audio


def rescale_audio(audio: np.ndarray):
    # Standardize type to be float32 [-1, 1]
    if audio.dtype.kind == 'i':
        max_val = max(np.iinfo(audio.dtype).max, -np.iinfo(audio.dtype).min)
        audio_scaled = audio.astype('float64') / max_val
    elif audio.dtype.kind == 'f':
        audio_scaled = audio.astype('float64')
    else:
        err_msg = 'Invalid audio dtype: {}'
        raise ValueError(err_msg.format(audio.dtype))

    # Map to the range [-2**31, 2**31]
    return (audio_scaled * (2**31)).astype('float32')


def normalize_loudness(audio: np.ndarray, sr: int,  target_db_lufs=-14.0, use_peak=True):
    if use_peak:
        return ln.normalize.peak(audio, target_db_lufs)
    else:
        meter = ln.Meter(sr)
        loudness = meter.integrated_loudness(audio)
        return ln.normalize.loudness(audio, loudness, target_db_lufs)