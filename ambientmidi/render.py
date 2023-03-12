import scipy
import scipy.signal
import numpy as np
import librosa
import librosa.core
import librosa.effects
from audiolazy.lazy_synth import adsr


def res_filter(audio, f, Q=45.0, sample_rate=16000):
    # Apply a resonance filter at a given frequency
    b, a = scipy.signal.iirpeak(f, Q=Q, fs=sample_rate)
    return scipy.signal.lfilter(b, a, audio)


def apply_adsr(audio, attack=0.03, decay=0.1, sustain=0.7, release=0.1):
    # Apply an Attack-Decay-Sustain-Release envelope to an audio signal
    duration = len(audio)
    assert 0 < attack + decay + release < 1
    attack = int(attack * duration)
    decay = int(decay * duration)
    release = int(release * duration)
    adsr_env = np.array(list(adsr(duration, attack, decay, sustain, release)))
    return audio * adsr_env


def apply_note_effects(
    audio, f0, harmonic_decay=0.5, num_harmonics=4, dominant_harmonic=0,
    resonance_quality=45.0, sample_rate=16000,
):
    assert 0 <= dominant_harmonic <= num_harmonics
    # Apply harmonic resonance filters to emphasize pitchness at
    # the corresponding fundamental
    # Harmonic weights follow a power series
    harmonic_weights = np.power(harmonic_decay, np.arange(num_harmonics) - dominant_harmonic)
    if dominant_harmonic > 0 and num_harmonics > 1:
        pre_dominant_weights = harmonic_weights[
            dominant_harmonic + 1 : min(num_harmonics, 2*dominant_harmonic + 1)
        ]
        if len(pre_dominant_weights) < dominant_harmonic:
            pre_dominant_weights = np.pad(
                pre_dominant_weights,
                (dominant_harmonic - len(pre_dominant_weights)),
                mode='constant',
            )
        harmonic_weights[:dominant_harmonic] = pre_dominant_weights
    filtered = np.zeros_like(audio)
    # Apply resonance filters
    for idx, weight in enumerate(harmonic_weights):
        n = idx + 1
        # Make sure harmonic doesn't exceed Nyquist
        if (f0 * n) < (sample_rate / 2):
            filtered += weight * res_filter(audio, f0 * n, Q=resonance_quality)
    return filtered
    

def render_song_from_events(song_dict, filtered_env_clusters_to_events, sample_rate=16000):
    print(f"Creating environmental song || {song_dict['name']}")
    duration_padding = 1.0
    playback_speed = 1.0
    velocity_factor = 5
    min_velocity_gain = 0.25
    drum_gain = 0.3
    cluster_instr_compare_key = 'mean_mfcc'
    cluster_feature_key = 'mfcc'
    apply_pitch_filtering = True

    duration = song_dict['duration'] + duration_padding
    num_samples = int(duration * sample_rate / playback_speed)
    audio_out = np.zeros((num_samples,))

    print(f"Arranging song")
    num_instrs = len(song_dict['instr_to_events'])
    for instr_idx, (instr_name, instr_item) in enumerate(song_dict['instr_to_events'].items()):
        print(f"* Instrument ({instr_idx+1}/{num_instrs}): {instr_name}")
        # If possible, only get matching drum or not-drum
        #filtered_env_clusters_to_events = {k: v for k, v in reduced_env_clusters_to_events.items()
        #                                  if v['is_drum'] == instr_item['is_drum']} or reduced_env_clusters_to_events
        # Pick the cluster with the closest feature centroid
        env_cluster_name, env_cluster_item = min(
            filtered_env_clusters_to_events.items(),
            key=lambda x: np.sqrt(
                (
                    (
                        x[1][cluster_instr_compare_key]
                        - instr_item[cluster_instr_compare_key]
                    ) ** 2
                ).sum()
            )
        )

        # Iterate through instrument notes:
        print(f"* Producing {len(instr_item['events'])} notes")
        for instr_event_dict in instr_item['events']:
            out_start_ts = instr_event_dict['midi_note']['start']
            out_start_idx = int(out_start_ts * sample_rate / playback_speed)
            # Get closest event in feature space
            env_event_dict = min(
                env_cluster_item['events'],
                key=lambda x: np.sqrt(
                    (
                        (
                            x[cluster_feature_key]
                            - instr_event_dict[cluster_feature_key]
                        ) ** 2
                    ).mean()
                )
            )

            if np.isfinite(instr_event_dict['midi_note']['velocity']):
                # Map velocity to gain via a shifted logarithmic mapping
                gain = (
                    np.log10(
                        1.0 + instr_event_dict['midi_note']['velocity'] / velocity_factor
                    ) + min_velocity_gain
                ) / (np.log10(1.0 + 127/velocity_factor) + min_velocity_gain)
            else:
                gain = 1.0
                
            if not instr_item['is_drum']:
                #### Pitched instrument ####
                note_audio = env_event_dict['audio']
                note_audio = note_audio / np.abs(note_audio).max()
                src_note_num = env_event_dict['pitch']
                dst_note_num = instr_event_dict['midi_note']['pitch']
                if np.isfinite(src_note_num):
                    # If a pitch was detected, then try to pitch shift
                    n_steps = dst_note_num - src_note_num
                    # Pitch shift event to correct pitch
                    note_audio = librosa.effects.pitch_shift(y=note_audio,
                                                            sr=sample_rate,
                                                            n_steps=n_steps)
                dst_note_f0 = librosa.core.midi_to_hz(dst_note_num)
                # Emphasize note harmonic frequencies
                if apply_pitch_filtering:
                    note_audio = apply_note_effects(note_audio, dst_note_f0,
                                harmonic_decay=0.7,
                                num_harmonics=3,
                                dominant_harmonic=0,
                                resonance_quality=10.0,
                                sample_rate=sample_rate)
                # Apply ADSR envelope
                note_audio = apply_adsr(
                    note_audio, attack=0.03, decay=0.1, sustain=0.7, release=0.1,
                )
            else:
                #### Drum instrument ####
                # Just apply drum gain and ADSR envelope
                note_audio = drum_gain * apply_adsr(
                    env_event_dict['audio'], 
                    attack=0.01, decay=0.02, sustain=0.7, release=0.02,
                )

            out_end_idx = out_start_idx + len(note_audio)

            # Add audio to mix
            audio_out[out_start_idx:out_end_idx] += gain * note_audio
            
    return audio_out