# AmbientMIDI v2.0

A robust, configurable MIDI synthesis system for creating ambient audio from MIDI files using recorded or live audio input.

## 🎵 What's New in v2.0

The AmbientMIDI system has been completely refactored to be more modular, robust, configurable, and high-quality:

### ✨ Key Improvements

- **🔧 Comprehensive Configuration System**: JSON-based configuration with validation and type safety
- **🏗️ Modular Pipeline Architecture**: Pluggable processing steps with clear interfaces
- **🛡️ Robust Error Handling**: Custom exception hierarchy with detailed error context
- **📊 Extensive Logging**: Configurable logging throughout the system
- **🎛️ Flexible Audio Processing**: Enhanced audio handling with multiple formats and options
- **📈 Progress Tracking**: Real-time progress updates during processing
- **💾 Intelligent Caching**: Automatic caching of processed MIDI data
- **🔍 Input Validation**: Comprehensive validation of all inputs and parameters
- **📋 Processing Summaries**: Detailed summaries of processing steps and timing

## 🚀 Quick Start

### Basic Usage

```python
from ambientmidi import process_midi_file
from pathlib import Path

# Simple processing with defaults
results = process_midi_file(
    midi_path="input.mid",
    output_path="output.wav"
)

# Check if processing succeeded
if all(result.success for result in results.values()):
    print("Processing completed successfully!")
else:
    print("Processing failed at some steps")
```

### Advanced Usage

```python
from ambientmidi import create_pipeline, AmbientMIDIConfig
from pathlib import Path

# Create custom configuration
config = AmbientMIDIConfig()
config.audio.sample_rate = 44100
config.audio.record_duration = 30.0
config.midi.samples_per_instrument = 15
config.paths.output_dir = Path("my_output")

# Create and run pipeline
pipeline = create_pipeline(config)

def progress_callback(step, progress):
    print(f"{step}: {progress:.1%}")

results = pipeline.process(
    midi_path=Path("input.mid"),
    output_path=Path("output.wav"),
    input_recording_path=Path("input_audio.wav"),  # Optional
    progress_callback=progress_callback
)

# Get processing summary
summary = pipeline.get_summary()
print(f"Total processing time: {summary['total_processing_time']:.2f}s")
```

## 🔧 Configuration

### Configuration File Example

Create a `config.json` file:

```json
{
  "audio": {
    "sample_rate": 16000,
    "record_duration": 60.0,
    "denoise_enabled": true,
    "use_peak_normalization": true,
    "target_db_lufs": -14.0,
    "min_clip_size_s": 0.125,
    "max_clip_size_s": 1.0
  },
  "midi": {
    "samples_per_instrument": 10,
    "soundfont_path": "/path/to/soundfont.sf2",
    "max_song_duration": 300.0
  },
  "render": {
    "harmonic_decay": 0.5,
    "num_harmonics": 4,
    "resonance_quality": 45.0,
    "attack": 0.03,
    "decay": 0.1,
    "sustain": 0.7,
    "release": 0.1
  },
  "paths": {
    "output_dir": "output",
    "cache_dir": "cache",
    "meta_dir": "meta",
    "temp_dir": "temp"
  },
  "logging": {
    "level": "INFO",
    "console_handler": true,
    "file_handler": "ambientmidi.log"
  }
}
```

### Loading Configuration

```python
from ambientmidi import load_config

# Load from file
config = load_config("config.json")

# Load from default locations
config = load_config()  # Checks ./config.json, ~/.ambientmidi/config.json, etc.

# Create with custom parameters
from ambientmidi import create_default_config

config = create_default_config(
    sample_rate=44100,
    record_duration=30.0,
    soundfont_path="/path/to/soundfont.sf2"
)
```

## 🏗️ Architecture

### Pipeline System

The processing pipeline consists of four main steps:

1. **MIDI Preprocessing**: Parse MIDI file and extract note events
2. **Audio Acquisition**: Record or load audio input
3. **Event Processing**: Detect onsets and cluster audio events
4. **Audio Rendering**: Generate final output audio

### Custom Processing Steps

You can create custom processing steps by extending the `ProcessingStep` class:

```python
from ambientmidi.pipeline import ProcessingStep, ProcessingResult
from ambientmidi.exceptions import ErrorHandler

class CustomProcessingStep(ProcessingStep):
    def __init__(self, config):
        super().__init__("custom_step", config)
    
    def process(self, input_data, **kwargs):
        start_time = time.time()
        
        try:
            with ErrorHandler("Custom processing"):
                # Your processing logic here
                result_data = self.custom_processing(input_data)
                
                return ProcessingResult(
                    data=result_data,
                    metadata={"custom_info": "value"},
                    processing_time=time.time() - start_time,
                    success=True
                )
        except Exception as e:
            return ProcessingResult(
                data=None,
                metadata={},
                processing_time=time.time() - start_time,
                success=False,
                error=e
            )
```

## 🎛️ Command Line Interface

```bash
# Basic usage
python synthesize.py input.mid output.wav

# With configuration file
python synthesize.py input.mid output.wav --config config.json

# With custom parameters
python synthesize.py input.mid output.wav \
    --sample-rate 44100 \
    --record-duration 30 \
    --samples-per-instrument 15 \
    --soundfont-path /path/to/soundfont.sf2 \
    --log-level DEBUG \
    --save-summary \
    --save-config

# With input recording
python synthesize.py input.mid output.wav \
    --input-recording input_audio.wav \
    --no-denoise \
    --verbose

# Show help
python synthesize.py --help
```

## 📊 Error Handling

The system includes comprehensive error handling with custom exception types:

```python
from ambientmidi.exceptions import (
    AmbientMIDIError,
    AudioProcessingError,
    MIDIProcessingError,
    ConfigurationError
)

try:
    results = process_midi_file("input.mid", "output.wav")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    print(f"Details: {e.details}")
except AudioProcessingError as e:
    print(f"Audio processing failed: {e}")
except MIDIProcessingError as e:
    print(f"MIDI processing failed: {e}")
except AmbientMIDIError as e:
    print(f"General AmbientMIDI error: {e}")
```

## 🧪 Testing and Validation

### Input Validation

All inputs are validated before processing:

```python
from ambientmidi.audio import validate_audio_array
from ambientmidi.exceptions import InvalidInputError

try:
    validate_audio_array(audio_data, "my audio processing")
except InvalidInputError as e:
    print(f"Invalid audio input: {e}")
```

### Configuration Validation

Configurations are automatically validated:

```python
from ambientmidi import AmbientMIDIConfig

config = AmbientMIDIConfig()
config.audio.sample_rate = -1  # Invalid value

try:
    config.validate_all()
except ValueError as e:
    print(f"Configuration validation failed: {e}")
```

## 📈 Performance and Monitoring

### Progress Tracking

```python
def detailed_progress_callback(step, progress):
    steps = {
        "midi_preprocessing": "Processing MIDI",
        "audio_acquisition": "Acquiring audio",
        "event_processing": "Processing events",
        "audio_rendering": "Rendering audio"
    }
    
    print(f"[{progress:.1%}] {steps.get(step, step)}")

results = process_midi_file(
    "input.mid",
    "output.wav",
    progress_callback=detailed_progress_callback
)
```

### Performance Monitoring

```python
# Get detailed processing summary
summary = pipeline.get_summary()
print(f"Total time: {summary['total_processing_time']:.2f}s")

for step_name, step_info in summary['steps'].items():
    print(f"{step_name}: {step_info['processing_time']:.2f}s")
    if step_info['error']:
        print(f"  Error: {step_info['error']}")
```

## 🔧 Advanced Features

### Caching

MIDI preprocessing results are automatically cached:

```python
# First run: processes and caches
results1 = process_midi_file("input.mid", "output1.wav")

# Second run: uses cached data
results2 = process_midi_file("input.mid", "output2.wav")
```

### Audio Processing Options

```python
from ambientmidi.audio import (
    load_audio,
    normalize_loudness,
    apply_fade,
    detect_silence
)

# Load and process audio
audio = load_audio("input.wav", sample_rate=16000)
audio = normalize_loudness(audio, 16000, target_db_lufs=-12.0)
audio = apply_fade(audio, fade_in_duration=0.5, fade_out_duration=0.5)

# Detect silence regions
silence_mask, silence_regions = detect_silence(audio, 16000)
```

### Event Processing

```python
from ambientmidi.events import (
    compute_pcengram,
    get_onsets,
    get_event_clip_dicts,
    filter_events_by_energy
)

# Process events
pcengram = compute_pcengram(audio, sample_rate=16000)
onset_idxs, onset_frames, onset_env = get_onsets(pcengram)
event_clips = get_event_clip_dicts(audio, onset_idxs)
filtered_clips = filter_events_by_energy(event_clips, energy_threshold=0.01)
```

## 🐛 Troubleshooting

### Common Issues

1. **Audio recording fails**: Check microphone permissions and PyAudio installation
2. **MIDI processing slow**: Reduce `samples_per_instrument` in configuration
3. **Out of memory**: Reduce `max_song_duration` or audio `record_duration`
4. **Poor quality output**: Increase `samples_per_instrument` or use better soundfont

### Debug Mode

```python
from ambientmidi import configure_package_logging
import logging

# Enable debug logging
configure_package_logging(logging.DEBUG)

# Now all operations will have detailed logging
results = process_midi_file("input.mid", "output.wav")
```

## 📋 Requirements

- Python 3.7+
- NumPy
- SciPy
- librosa
- soundfile
- PyAudio
- resampy
- pretty_midi
- noisereduce
- pyloudnorm
- scikit-learn-extra
- pyfluidsynth

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper error handling and logging
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- librosa team for audio processing tools
- pretty_midi team for MIDI handling
- Contributors to the original AmbientMIDI codebase

---

For more examples and detailed API documentation, see the `examples/` directory and docstrings in the code.