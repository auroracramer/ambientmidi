import os
import resampy
import pyaudio
import numpy as np
import soundfile as sf
from constants import SAMPLE_RATE
from lakh import LakhMIDIDataset
from music import create_song

CHUNKSIZE = 1024

def run(lakh_data_dir, recording_duration_seconds=200.0, num_songs=None):
    # Load MIDI dataset
    print("Loading Lakh MIDI dataset")
    midi_dataset = LakhMIDIDataset(lakh_data_dir, max_num_songs=num_songs)

    # Record audio
    p = pyaudio.PyAudio()
    tmp_path = 'tmp_audio.wav'
    if os.path.exists(tmp_path):
        inp_audio, _ = sf.read(tmp_path)
    else:
        print(f"Recording {recording_duration_seconds:0.2f} seconds of audio")
        num_input_samples = int(recording_duration_seconds * SAMPLE_RATE)

        inp_stream = p.open(format=pyaudio.paFloat32,
                            channels=1,
                            rate=SAMPLE_RATE,
                            input=True)
        inp_audio = inp_stream.read(num_input_samples)
        inp_audio = np.frombuffer(inp_audio, dtype=np.float32)
        inp_stream.stop_stream()
        inp_stream.close()
        sf.write(tmp_path, inp_audio, SAMPLE_RATE)

    # Create song
    print(f"Creating song")
    out_audio = create_song(inp_audio, midi_dataset)
    sf.write("song.wav", out_audio, SAMPLE_RATE)

    ## Play output audio
    #print(f"Playing song")

    #device_sample_rate = int(p.get_default_output_device_info().get('defaultSampleRate'))
    #if device_sample_rate != SAMPLE_RATE:
    #    stream_audio = resampy.resample(out_audio, SAMPLE_RATE, device_sample_rate)
    #else:
    #    stream_audio = out_audio
    #out_stream = p.open(format=pyaudio.paFloat32,
    #                    channels=1,
    #                    rate=device_sample_rate,
    #                    output=True)
    #out_stream.write(stream_audio.astype(np.float32).tostring())
    #out_stream.stop_stream()
    #out_stream.close()
    #p.terminate()


if __name__ == '__main__':
    lakh_data_dir = '/home/jsondotload/data/lmd'
    recording_duration_seconds = 60.0#200.0
    num_songs = 40
    run(lakh_data_dir,
        recording_duration_seconds=recording_duration_seconds,
        num_songs=num_songs)