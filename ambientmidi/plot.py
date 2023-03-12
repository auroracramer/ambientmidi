import numpy as np
import librosa
import matplotlib.pyplot as plt


def plot_events(pcengram, onset_env, onset_frames, n_fft, hop_length, sample_rate):
    times = librosa.times_like(onset_env, sr=sample_rate, hop_length=hop_length, n_fft=n_fft)

    #D = librosa.amplitude_to_db(np.abs(librosa.stft(inp_audio_scaled)), ref=np.max)
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    img = librosa.display.specshow(pcengram, y_axis='log', x_axis='s',
                                hop_length=hop_length,
                                sr=sample_rate, ax=axes[0])
    axes[0].set_title('Power spectrogram')
    axes[0].label_outer()
    axes[1].plot(times, onset_env, alpha=0.8, label='Onset strength')
    axes[1].vlines(
        times[onset_frames], 0, onset_env.max(),
        color='r', alpha=0.9, linestyle='--', label='Onsets',
    )
    axes[1].legend()
    axes[1].set_ylabel('Onset strength')
    return fig