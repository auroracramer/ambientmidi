"""
AmbientMIDI - A robust, configurable MIDI synthesis system

This package provides a modular system for creating ambient audio
from MIDI files using recorded or live audio input.

Main components:
- Configuration system for all aspects of processing
- Pipeline system for orchestrating the workflow
- Audio processing with robust error handling
- Event detection and clustering
- MIDI preprocessing and synthesis
- Audio rendering with effects

Example usage:
    from ambientmidi import AmbientMIDIConfig, create_pipeline
    from pathlib import Path
    
    # Create configuration
    config = AmbientMIDIConfig()
    
    # Create and run pipeline
    pipeline = create_pipeline(config)
    results = pipeline.process(
        midi_path=Path("input.mid"),
        output_path=Path("output.wav")
    )
"""

# Version information
__version__ = "2.0.0"
__author__ = "AmbientMIDI Team"
__email__ = "ambientmidi@example.com"

# Core configuration system
from .config import (
    AmbientMIDIConfig,
    AudioConfig,
    SpectrogramConfig,
    MIDIConfig,
    RenderConfig,
    ClusteringConfig,
    PathConfig,
    LoggingConfig,
    LogLevel,
    get_default_config,
    load_config
)

# Pipeline system
from .pipeline import (
    AmbientMIDIPipeline,
    ProcessingStep,
    ProcessingResult,
    MIDIPreprocessingStep,
    AudioAcquisitionStep,
    EventProcessingStep,
    AudioRenderingStep,
    create_pipeline
)

# Exception system
from .exceptions import (
    AmbientMIDIError,
    ConfigurationError,
    AudioProcessingError,
    MIDIProcessingError,
    RenderingError,
    ClusteringError,
    FeatureExtractionError,
    FileNotFoundError,
    InvalidInputError,
    ProcessingTimeoutError,
    AudioRecordingError,
    SoundFontError,
    OnsetDetectionError,
    SpectrogramError,
    ErrorHandler,
    handle_error
)

# Audio processing
from .audio import (
    record_audio,
    load_audio,
    rescale_audio,
    normalize_loudness,
    apply_fade,
    detect_silence,
    validate_audio_array,
    load_and_prepare_audio,
    record_and_prepare_audio
)

# Event processing
from .events import (
    compute_pcengram,
    get_onsets,
    get_event_clip_dicts,
    filter_events_by_energy,
    compute_onset_strength_envelope,
    truncate_silence,
    compute_pcengram_from_config,
    get_onsets_from_config,
    get_event_clips_from_config
)

# MIDI processing
from .midi import (
    preprocess_midi_file,
    # Note: other midi functions would be imported here if refactored
)

# Audio rendering
from .render import (
    render_song_from_events,
    res_filter,
    apply_adsr,
    apply_note_effects,
    # Note: other render functions would be imported here if refactored
)

# Clustering
from .cluster import (
    get_clip_clusters,
    # Note: other clustering functions would be imported here if refactored
)

# Feature extraction
from .features import (
    get_feature_dict,
    # Note: other feature functions would be imported here if refactored
)

# Utility functions
from .utils import (
    LazyDict,
    qtile,
    NpEncoder
)

# Convenience functions for common workflows
def process_midi_file(
    midi_path,
    output_path,
    input_recording_path=None,
    config=None,
    progress_callback=None
):
    """
    Process a MIDI file with optional audio input.
    
    This is a convenience function that creates a pipeline and processes
    a MIDI file in one step.
    
    Args:
        midi_path: Path to MIDI file
        output_path: Path for output audio file
        input_recording_path: Optional path to input recording
        config: Optional configuration object
        progress_callback: Optional progress callback function
        
    Returns:
        Dictionary of processing results
    """
    from pathlib import Path
    
    # Convert to Path objects
    midi_path = Path(midi_path)
    output_path = Path(output_path)
    if input_recording_path:
        input_recording_path = Path(input_recording_path)
    
    # Use default config if none provided
    if config is None:
        config = get_default_config()
    
    # Create and run pipeline
    pipeline = create_pipeline(config)
    results = pipeline.process(
        midi_path=midi_path,
        output_path=output_path,
        input_recording_path=input_recording_path,
        progress_callback=progress_callback
    )
    
    return results


def create_default_config(
    sample_rate=16000,
    record_duration=60.0,
    samples_per_instrument=10,
    soundfont_path=None,
    output_dir="output",
    cache_dir="cache"
):
    """
    Create a default configuration with custom parameters.
    
    Args:
        sample_rate: Audio sample rate
        record_duration: Recording duration in seconds
        samples_per_instrument: Number of samples per instrument
        soundfont_path: Path to soundfont file
        output_dir: Output directory
        cache_dir: Cache directory
        
    Returns:
        AmbientMIDIConfig object
    """
    config = get_default_config()
    
    # Update with custom parameters
    config.audio.sample_rate = sample_rate
    config.audio.record_duration = record_duration
    config.midi.samples_per_instrument = samples_per_instrument
    
    if soundfont_path:
        from pathlib import Path
        config.midi.soundfont_path = Path(soundfont_path)
    
    from pathlib import Path
    config.paths.output_dir = Path(output_dir)
    config.paths.cache_dir = Path(cache_dir)
    
    # Re-validate configuration
    config.validate_all()
    
    return config


def get_version_info():
    """
    Get version and package information.
    
    Returns:
        Dictionary with version information
    """
    return {
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "description": "A robust, configurable MIDI synthesis system"
    }


# Define public API
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    
    # Configuration
    "AmbientMIDIConfig",
    "AudioConfig",
    "SpectrogramConfig",
    "MIDIConfig",
    "RenderConfig",
    "ClusteringConfig",
    "PathConfig",
    "LoggingConfig",
    "LogLevel",
    "get_default_config",
    "load_config",
    
    # Pipeline
    "AmbientMIDIPipeline",
    "ProcessingStep",
    "ProcessingResult",
    "MIDIPreprocessingStep",
    "AudioAcquisitionStep",
    "EventProcessingStep",
    "AudioRenderingStep",
    "create_pipeline",
    
    # Exceptions
    "AmbientMIDIError",
    "ConfigurationError",
    "AudioProcessingError",
    "MIDIProcessingError",
    "RenderingError",
    "ClusteringError",
    "FeatureExtractionError",
    "FileNotFoundError",
    "InvalidInputError",
    "ProcessingTimeoutError",
    "AudioRecordingError",
    "SoundFontError",
    "OnsetDetectionError",
    "SpectrogramError",
    "ErrorHandler",
    "handle_error",
    
    # Audio processing
    "record_audio",
    "load_audio",
    "rescale_audio",
    "normalize_loudness",
    "apply_fade",
    "detect_silence",
    "validate_audio_array",
    "load_and_prepare_audio",
    "record_and_prepare_audio",
    
    # Event processing
    "compute_pcengram",
    "get_onsets",
    "get_event_clip_dicts",
    "filter_events_by_energy",
    "compute_onset_strength_envelope",
    "truncate_silence",
    "compute_pcengram_from_config",
    "get_onsets_from_config",
    "get_event_clips_from_config",
    
    # MIDI processing
    "preprocess_midi_file",
    
    # Audio rendering
    "render_song_from_events",
    "res_filter",
    "apply_adsr",
    "apply_note_effects",
    
    # Clustering
    "get_clip_clusters",
    
    # Feature extraction
    "get_feature_dict",
    
    # Utilities
    "LazyDict",
    "qtile",
    "NpEncoder",
    
    # Convenience functions
    "process_midi_file",
    "create_default_config",
    "get_version_info",
]


# Set up logging for the package
import logging

# Create package logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add null handler to prevent logging errors if no handlers are configured
logger.addHandler(logging.NullHandler())

# Log package initialization
logger.debug(f"AmbientMIDI v{__version__} initialized")


# Optional: Set up package-level configuration
def configure_package_logging(level=logging.INFO, format_string=None):
    """
    Configure logging for the entire package.
    
    Args:
        level: Logging level
        format_string: Optional custom format string
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger for the package
    package_logger = logging.getLogger(__name__)
    package_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in package_logger.handlers[:]:
        package_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(format_string)
    console_handler.setFormatter(formatter)
    package_logger.addHandler(console_handler)
    
    logger.info(f"Package logging configured at level {logging.getLevelName(level)}")


# Make configuration easily accessible
# DEFAULT_CONFIG = get_default_config()  # Commented out to avoid dependency issues