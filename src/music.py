import os
import librosa
import random
import numpy as np
import pretty_midi
import pickle as pk
from env_processing import extract_clusters_to_events
from constants import SAMPLE_RATE

DURATION_PADDING = 1.0

def create_song(input_audio, midi_dataset):
    # Select song
    song_name, song_dict = random.choice(list(midi_dataset.songs.items()))
    print(f"Creating environmental song || {song_name}")

    # Get event clusters from recording
    tmp_path = "clusters.pkl"
    if os.path.exists(tmp_path):
        with open(tmp_path, 'rb') as f:
            env_cluster_to_events = pk.load(f)
    else:
        env_cluster_to_events = extract_clusters_to_events(input_audio, SAMPLE_RATE)
        with open(tmp_path, 'wb') as f:
            pk.dump(env_cluster_to_events, f)

    duration = song_dict['duration'] + DURATION_PADDING
    num_samples = int(duration * SAMPLE_RATE)
    audio_out = np.zeros((num_samples,))

    print(f"Arranging song")
    num_instrs = len(song_dict['instr_to_events'])
    for instr_idx, (instr_name, instr_item) in enumerate(song_dict['instr_to_events'].items()):
        print(f"* Instrument ({instr_idx+1}/{num_instrs}): {instr_name}")
        # If possible, only get matching drum or not-drum
        filtered_env_cluster_to_events = {k: v for k, v in env_cluster_to_events.items()
                                          if v['is_drum'] == instr_item['is_drum']} or env_cluster_to_events
        # Pick the cluster with the closest centroid
        env_cluster_name, env_cluster_item = min(
            filtered_env_cluster_to_events.items(),
            key=lambda x: np.sqrt(((x[1]['mean_mfcc'] - instr_item['mean_mfcc']) ** 2).sum()))

        # Iterate through instrument notes:
        print(f"* Producing {len(instr_item['events'])} notes")
        for instr_event_dict in instr_item['events']:
            out_start_ts = instr_event_dict['midi_note']['start']
            out_start_idx = int(out_start_ts * SAMPLE_RATE)

            if not instr_item['is_drum']:
                # Get closest event in terms of MFCC and pitch
                #env_event_dict = min(env_cluster_item['events'],
                #                     key=lambda x: np.sqrt(((x['mfcc'] - instr_event_dict['mfcc']) ** 2).mean())
                #                                   + np.sqrt(x['pitch_hz'] - (instr_event_dict['midi_note']['pitch_hz'] or pretty_midi.note_number_to_hz(instr_event_dict['midi_note']['pitch']))))
                env_event_dict = random.choice(env_cluster_item['events'])
                note_audio = env_event_dict['audio']
                src_note_num = env_event_dict['pitch']
                if np.isfinite(src_note_num):
                    # If a pitch was detected, then try to pitch shift
                    dst_note_num = instr_event_dict['midi_note']['pitch']
                    n_steps = dst_note_num - src_note_num
                    # Pitch shift event to correct pitch
                    note_audio = librosa.effects.pitch_shift(y=note_audio,
                                                             sr=SAMPLE_RATE,
                                                             n_steps=n_steps)
            else:
                env_event_dict = min(env_cluster_item['events'],
                                     key=lambda x: np.sqrt(((x['mfcc'] - instr_event_dict['mfcc']) ** 2).mean()))
                note_audio = env_event_dict['audio']
            # TODO: More effects! Maybe HPSS

            out_end_idx = out_start_idx + len(note_audio)

            # Add audio to mix
            audio_out[out_start_idx:out_end_idx] = note_audio

    return audio_out
