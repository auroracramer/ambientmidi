import pretty_midi
import numpy as np
from ambientmidi.features import get_feature_dict
from pathlib import Path


def preprocess_midi_file(midi_path, sample_rate=16000, samples_per_instr=10, soundfont_path=None):
    pm = pretty_midi.PrettyMIDI(midi_path)
    instr_to_events = {}

    print(f"           ~ collecting notes")
    for instr in pm.instruments:
        for note in instr.notes:
            # Get instrument name
            if instr.is_drum:
                instr_name = f"{instr.name} - {pretty_midi.note_number_to_drum_name(note.pitch)}"
            else:
                instr_name = instr.name

            if instr_name not in instr_to_events:
                instr_to_events[instr_name] = {
                    'events': []
                }

            # Get information about MIDI note
            note_dict = {'program': instr.program,
                        'name': instr_name,
                        'is_drum': instr.is_drum,
                        'velocity': note.velocity,
                        'pitch': note.pitch,
                        'pitch_hz': pretty_midi.note_number_to_hz(note.pitch) if not instr.is_drum else None,
                        'start': note.start,
                        'end': note.end,
                        'duration': note.get_duration()}

            event_dict = {'midi_note': note_dict}
            instr_to_events[instr_name]['events'].append(event_dict)

    for instr_name, cluster_item in instr_to_events.items():
        X_midi = np.array([[event['midi_note']['velocity'] if not np.isnan(event['midi_note']['velocity']) else 127,
                            event['midi_note']['pitch'],
                            event['midi_note']['duration']]
                        for event in cluster_item['events']])
        if X_midi.shape[0] <= samples_per_instr:
            X_midi_samples = X_midi
            group_to_event_idxs = [[event_idx]
                                for event_idx in range(X_midi.shape[0])]
        else:
            # Group note events as a shortcut to save on synthesis
            # and feature extraction time
            print(f"           ~ {instr_name}: grouping note events")
            X_midi_mean = X_midi.mean(axis=0, keepdims=True)
            X_midi_std = X_midi.std(axis=0, keepdims=True)
            X_midi_pp = (X_midi - X_midi_mean) / X_midi_std
            X_midi_pp[np.logical_not(np.isfinite(X_midi_pp))] = 0.0
            kmedoids = KMedoids(n_clusters=samples_per_instr)
            note_group_idxs = kmedoids.fit_predict(X_midi_pp)
            group_to_event_idxs = [[] for _ in range(samples_per_instr)]
            for event_idx, group_idx in enumerate(note_group_idxs):
                group_to_event_idxs[group_idx].append(event_idx)
            X_midi_samples = kmedoids.cluster_centers_ * X_midi_std + X_midi_mean
        program_name = cluster_item['events'][0]['midi_note']['program']
        print(f"           ~ {instr_name}: synthesizing and computing audio features")
        for group_idx, (velocity, pitch, duration) in enumerate(X_midi_samples):
            # Create a dummy program to synthesize note audio
            note_pm = pretty_midi.PrettyMIDI()
            note_instr = pretty_midi.Instrument(program=program_name)
            note_pm.instruments.append(note_instr)
            note_instr.notes.append(pretty_midi.Note(velocity=int(velocity), pitch=int(pitch), start=0.0, end=duration))
            note_audio = note_pm.fluidsynth(fs=sample_rate,
                                            sf2_path=soundfont_path)
            # Compute audio features from synthesized audio
            feature_dict = get_feature_dict(note_audio, sample_rate, no_audio=True)
            for event_idx in group_to_event_idxs[group_idx]:
                cluster_item['events'][event_idx].update(feature_dict)

    print(f"           ~ computing instrument features")
    for instr_name, cluster_item in instr_to_events.items():
        X_mfcc = np.array([x['mfcc'] for x in cluster_item['events']])
        X_tonality = np.array([x['tonality'] for x in cluster_item['events']])
        cluster_item['mean_mfcc'] = X_mfcc.mean()
        cluster_item['mean_tonality'] = X_tonality.mean()
        # is_drum is most common value of is_drum for each note for robustness
        is_drum_lst = [x['midi_note']['is_drum'] for x in cluster_item['events']]
        cluster_item['is_drum'] = max(set(is_drum_lst), key=is_drum_lst.count)
        cluster_item['num_events'] = len(cluster_item['events'])

    return {
        'name': Path(midi_path).stem,
        'duration': pm.get_end_time(),
        'instr_to_events': instr_to_events,
    }
