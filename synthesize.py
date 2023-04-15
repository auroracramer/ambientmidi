import json
import sys
import soundfile as sf
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional
from ambientmidi.midi import preprocess_midi_file
from ambientmidi.audio import load_audio, record_audio
from ambientmidi.events import compute_pcengram, get_event_clip_dicts, get_onsets
from ambientmidi.cluster import get_clip_clusters
from ambientmidi.render import render_song_from_events
from ambientmidi.utils import NpEncoder


def parse_args(args):
    p = ArgumentParser()
    p.add_argument("midi_path", type=Path)
    p.add_argument("output_path", type=Path)
    p.add_argument("meta_dir", type=Path)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--input_recording_path", nargs="?", type=Path)
    p.add_argument("--record_duration", type=int, default=60)
    p.add_argument("--midi_samples_per_instr", type=int, default=10)
    p.add_argument("--soundfont_path", nargs="?", type=Path)

    return vars(p.parse_args(args))


def main(
    midi_path: Path, output_path: Path, meta_dir: Path, sample_rate: int,
    record_duration: int, midi_samples_per_instr: int,
    input_recording_path: Optional[Path], soundfont_path: Optional[Path],
):
    # Preprocess input MIDI file to get note events and to get samples for each instrument
    meta_dir.mkdir(exist_ok=True, parents=True)
    meta_path = meta_dir.joinpath(f"{midi_path.stem}.json")
    if not meta_path.exists():
        midi_info = preprocess_midi_file(str(midi_path), sample_rate, midi_samples_per_instr, soundfont_path)
        with meta_path.open("w") as f:
            json.dump(midi_info, f, cls=NpEncoder)
    else:
        with meta_path.open("r") as f:
            midi_info = json.load(f)
        
    if input_recording_path:
        audio = load_audio(input_recording_path, sample_rate)
    else:
        audio = record_audio(record_duration, sample_rate)

    pcengram = compute_pcengram(audio, sample_rate=sample_rate)
    onset_idxs, onset_frames, onset_env = get_onsets(pcengram, sample_rate=sample_rate)

    event_clip_list = get_event_clip_dicts(audio, onset_idxs, sample_rate=sample_rate)
    env_clusters_to_events = get_clip_clusters(event_clip_list)
    audio_out = render_song_from_events(midi_info, env_clusters_to_events)

    sf.write(str(output_path), audio_out, sample_rate)


if __name__ == "__main__":
    arg_dict = parse_args(sys.argv[1:])
    main(**arg_dict)