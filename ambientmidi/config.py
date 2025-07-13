"""
Configuration management for AmbientMIDI.

This module provides a comprehensive configuration system with validation,
defaults, and type safety for all aspects of the MIDI synthesis pipeline.
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from enum import Enum


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 16000
    target_db_lufs: float = -14.0
    use_peak_normalization: bool = True
    denoise_enabled: bool = True
    record_duration: float = 60.0
    min_clip_size_s: float = 0.125
    max_clip_size_s: float = 1.0
    
    def __post_init__(self):
        self.validate()
    
    def validate(self) -> None:
        """Validate audio configuration parameters."""
        if self.sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {self.sample_rate}")
        if self.record_duration <= 0:
            raise ValueError(f"Record duration must be positive, got {self.record_duration}")
        if self.min_clip_size_s <= 0 or self.max_clip_size_s <= 0:
            raise ValueError("Clip sizes must be positive")
        if self.min_clip_size_s >= self.max_clip_size_s:
            raise ValueError("Min clip size must be less than max clip size")


@dataclass
class SpectrogramConfig:
    """Spectrogram processing configuration."""
    window_size_ms: float = 25.0
    hop_size_ms: float = 10.0
    n_mels: int = 40
    power: float = 2.0
    
    @property
    def n_fft(self) -> int:
        """Calculate n_fft based on window size and sample rate."""
        return int((self.window_size_ms / 1000) * AudioConfig().sample_rate)
    
    @property
    def hop_length(self) -> int:
        """Calculate hop length based on hop size and sample rate."""
        return int((self.hop_size_ms / 1000) * AudioConfig().sample_rate)
    
    def validate(self) -> None:
        """Validate spectrogram configuration parameters."""
        if self.window_size_ms <= 0 or self.hop_size_ms <= 0:
            raise ValueError("Window and hop sizes must be positive")
        if self.n_mels <= 0:
            raise ValueError(f"Number of mel bands must be positive, got {self.n_mels}")
        if self.power <= 0:
            raise ValueError(f"Power must be positive, got {self.power}")


@dataclass
class MIDIConfig:
    """MIDI processing configuration."""
    samples_per_instrument: int = 10
    soundfont_path: Optional[Path] = None
    max_song_duration: Optional[float] = None
    velocity_range: tuple = (1, 127)
    pitch_range: tuple = (0, 127)
    
    def __post_init__(self):
        self.validate()
    
    def validate(self) -> None:
        """Validate MIDI configuration parameters."""
        if self.samples_per_instrument <= 0:
            raise ValueError(f"Samples per instrument must be positive, got {self.samples_per_instrument}")
        if self.soundfont_path and not Path(self.soundfont_path).exists():
            raise FileNotFoundError(f"Soundfont file not found: {self.soundfont_path}")
        if self.max_song_duration is not None and self.max_song_duration <= 0:
            raise ValueError(f"Max song duration must be positive, got {self.max_song_duration}")


@dataclass
class RenderConfig:
    """Audio rendering configuration."""
    harmonic_decay: float = 0.5
    num_harmonics: int = 4
    dominant_harmonic: int = 0
    resonance_quality: float = 45.0
    attack: float = 0.03
    decay: float = 0.1
    sustain: float = 0.7
    release: float = 0.1
    
    def __post_init__(self):
        self.validate()
    
    def validate(self) -> None:
        """Validate render configuration parameters."""
        if not (0 <= self.harmonic_decay <= 1):
            raise ValueError(f"Harmonic decay must be between 0 and 1, got {self.harmonic_decay}")
        if self.num_harmonics <= 0:
            raise ValueError(f"Number of harmonics must be positive, got {self.num_harmonics}")
        if not (0 <= self.dominant_harmonic <= self.num_harmonics):
            raise ValueError(f"Dominant harmonic must be between 0 and {self.num_harmonics}")
        if self.resonance_quality <= 0:
            raise ValueError(f"Resonance quality must be positive, got {self.resonance_quality}")
        
        # Validate ADSR envelope
        adsr_sum = self.attack + self.decay + self.release
        if not (0 < adsr_sum < 1):
            raise ValueError(f"ADSR envelope (attack+decay+release) must be between 0 and 1, got {adsr_sum}")


@dataclass
class ClusteringConfig:
    """Clustering configuration."""
    algorithm: str = "kmedoids"
    random_state: int = 42
    max_iter: int = 300
    tol: float = 1e-4
    
    def validate(self) -> None:
        """Validate clustering configuration parameters."""
        if self.algorithm not in ["kmedoids", "kmeans"]:
            raise ValueError(f"Unsupported clustering algorithm: {self.algorithm}")
        if self.max_iter <= 0:
            raise ValueError(f"Max iterations must be positive, got {self.max_iter}")
        if self.tol <= 0:
            raise ValueError(f"Tolerance must be positive, got {self.tol}")


@dataclass
class PathConfig:
    """Path configuration."""
    output_dir: Path = Path("output")
    cache_dir: Path = Path("cache")
    meta_dir: Path = Path("meta")
    temp_dir: Path = Path("temp")
    
    def __post_init__(self):
        self.validate()
        self.create_directories()
    
    def validate(self) -> None:
        """Validate path configuration."""
        # Convert strings to Path objects if necessary
        for attr_name in ["output_dir", "cache_dir", "meta_dir", "temp_dir"]:
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, str):
                setattr(self, attr_name, Path(attr_value))
    
    def create_directories(self) -> None:
        """Create all configured directories."""
        for path in [self.output_dir, self.cache_dir, self.meta_dir, self.temp_dir]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler: Optional[Path] = None
    console_handler: bool = True
    
    def configure_logging(self) -> None:
        """Configure the logging system."""
        # Clear existing handlers
        logger = logging.getLogger()
        logger.handlers.clear()
        
        # Set level
        logger.setLevel(getattr(logging, self.level.value))
        
        # Create formatter
        formatter = logging.Formatter(self.format)
        
        # Console handler
        if self.console_handler:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if self.file_handler:
            file_handler = logging.FileHandler(self.file_handler)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)


@dataclass
class AmbientMIDIConfig:
    """Main configuration class that combines all sub-configurations."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    spectrogram: SpectrogramConfig = field(default_factory=SpectrogramConfig)
    midi: MIDIConfig = field(default_factory=MIDIConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        """Initialize logging and validate all configurations."""
        self.logging.configure_logging()
        self.validate_all()
    
    def validate_all(self) -> None:
        """Validate all configuration sections."""
        self.audio.validate()
        self.spectrogram.validate()
        self.midi.validate()
        self.render.validate()
        self.clustering.validate()
        # paths validation is handled in __post_init__
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AmbientMIDIConfig":
        """Create configuration from dictionary."""
        config = cls()
        
        # Update audio config
        if "audio" in config_dict:
            for key, value in config_dict["audio"].items():
                if hasattr(config.audio, key):
                    setattr(config.audio, key, value)
        
        # Update other configs similarly
        for section_name in ["spectrogram", "midi", "render", "clustering", "paths", "logging"]:
            if section_name in config_dict:
                section = getattr(config, section_name)
                for key, value in config_dict[section_name].items():
                    if hasattr(section, key):
                        setattr(section, key, value)
        
        config.validate_all()
        return config
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "AmbientMIDIConfig":
        """Load configuration from JSON file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def save(self, config_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    def get_cache_path(self, midi_path: Path) -> Path:
        """Get cache path for a MIDI file."""
        return self.paths.cache_dir / f"{midi_path.stem}.json"
    
    def get_output_path(self, midi_path: Path, suffix: str = "") -> Path:
        """Get output path for a processed file."""
        stem = midi_path.stem
        if suffix:
            stem += f"_{suffix}"
        return self.paths.output_dir / f"{stem}.wav"


def get_default_config() -> AmbientMIDIConfig:
    """Get default configuration."""
    return AmbientMIDIConfig()


def load_config(config_path: Optional[Union[str, Path]] = None) -> AmbientMIDIConfig:
    """Load configuration from file or return default."""
    if config_path is None:
        # Try to load from default locations
        default_paths = [
            Path("config.json"),
            Path("~/.ambientmidi/config.json").expanduser(),
            Path("/etc/ambientmidi/config.json")
        ]
        
        for path in default_paths:
            if path.exists():
                config_path = path
                break
    
    if config_path and Path(config_path).exists():
        return AmbientMIDIConfig.from_file(config_path)
    else:
        return get_default_config()