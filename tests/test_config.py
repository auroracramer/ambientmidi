"""
Unit tests for the AmbientMIDI configuration system.

Tests cover:
- Configuration creation and validation
- Serialization and deserialization
- Error handling and edge cases
- Configuration merging and updates
- Path management and validation
"""

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from ambientmidi.config import (
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
from tests import TestFixtures


class TestAudioConfig(unittest.TestCase):
    """Test AudioConfig class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fixtures = TestFixtures()
        
    def test_default_creation(self):
        """Test creating AudioConfig with default values."""
        config = AudioConfig()
        
        self.assertEqual(config.sample_rate, 16000)
        self.assertEqual(config.target_db_lufs, -14.0)
        self.assertTrue(config.use_peak_normalization)
        self.assertTrue(config.denoise_enabled)
        self.assertEqual(config.record_duration, 60.0)
        self.assertEqual(config.min_clip_size_s, 0.125)
        self.assertEqual(config.max_clip_size_s, 1.0)
    
    def test_custom_creation(self):
        """Test creating AudioConfig with custom values."""
        config = AudioConfig(
            sample_rate=44100,
            target_db_lufs=-20.0,
            use_peak_normalization=False,
            denoise_enabled=False,
            record_duration=30.0,
            min_clip_size_s=0.1,
            max_clip_size_s=2.0
        )
        
        self.assertEqual(config.sample_rate, 44100)
        self.assertEqual(config.target_db_lufs, -20.0)
        self.assertFalse(config.use_peak_normalization)
        self.assertFalse(config.denoise_enabled)
        self.assertEqual(config.record_duration, 30.0)
        self.assertEqual(config.min_clip_size_s, 0.1)
        self.assertEqual(config.max_clip_size_s, 2.0)
    
    def test_validation_positive_sample_rate(self):
        """Test validation of positive sample rate."""
        with self.assertRaises(ValueError) as cm:
            AudioConfig(sample_rate=0)
        self.assertIn("Sample rate must be positive", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            AudioConfig(sample_rate=-1000)
        self.assertIn("Sample rate must be positive", str(cm.exception))
    
    def test_validation_positive_record_duration(self):
        """Test validation of positive record duration."""
        with self.assertRaises(ValueError) as cm:
            AudioConfig(record_duration=0)
        self.assertIn("Record duration must be positive", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            AudioConfig(record_duration=-5.0)
        self.assertIn("Record duration must be positive", str(cm.exception))
    
    def test_validation_clip_sizes(self):
        """Test validation of clip size parameters."""
        # Test negative clip sizes
        with self.assertRaises(ValueError) as cm:
            AudioConfig(min_clip_size_s=-0.1)
        self.assertIn("Clip sizes must be positive", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            AudioConfig(max_clip_size_s=0)
        self.assertIn("Clip sizes must be positive", str(cm.exception))
        
        # Test min >= max
        with self.assertRaises(ValueError) as cm:
            AudioConfig(min_clip_size_s=1.0, max_clip_size_s=0.5)
        self.assertIn("Min clip size must be less than max clip size", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            AudioConfig(min_clip_size_s=1.0, max_clip_size_s=1.0)
        self.assertIn("Min clip size must be less than max clip size", str(cm.exception))


class TestSpectrogramConfig(unittest.TestCase):
    """Test SpectrogramConfig class."""
    
    def test_default_creation(self):
        """Test creating SpectrogramConfig with default values."""
        config = SpectrogramConfig()
        
        self.assertEqual(config.window_size_ms, 25.0)
        self.assertEqual(config.hop_size_ms, 10.0)
        self.assertEqual(config.n_mels, 40)
        self.assertEqual(config.power, 2.0)
    
    def test_computed_properties(self):
        """Test computed n_fft and hop_length properties."""
        config = SpectrogramConfig(window_size_ms=25.0, hop_size_ms=10.0)
        
        # With default sample rate of 16000
        expected_n_fft = int((25.0 / 1000) * 16000)  # 400
        expected_hop_length = int((10.0 / 1000) * 16000)  # 160
        
        self.assertEqual(config.n_fft, expected_n_fft)
        self.assertEqual(config.hop_length, expected_hop_length)
    
    def test_validation(self):
        """Test validation of spectrogram parameters."""
        # Test positive window and hop sizes
        with self.assertRaises(ValueError) as cm:
            config = SpectrogramConfig(window_size_ms=0)
            config.validate()
        self.assertIn("Window and hop sizes must be positive", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            config = SpectrogramConfig(hop_size_ms=-5.0)
            config.validate()
        self.assertIn("Window and hop sizes must be positive", str(cm.exception))
        
        # Test positive n_mels
        with self.assertRaises(ValueError) as cm:
            config = SpectrogramConfig(n_mels=0)
            config.validate()
        self.assertIn("Number of mel bands must be positive", str(cm.exception))
        
        # Test positive power
        with self.assertRaises(ValueError) as cm:
            config = SpectrogramConfig(power=0)
            config.validate()
        self.assertIn("Power must be positive", str(cm.exception))


class TestMIDIConfig(unittest.TestCase):
    """Test MIDIConfig class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = TestFixtures.create_temp_dir()
        self.soundfont_path = self.temp_dir / "test.sf2"
        self.soundfont_path.touch()  # Create empty file
    
    def tearDown(self):
        """Clean up test fixtures."""
        TestFixtures.cleanup_temp_dir(self.temp_dir)
    
    def test_default_creation(self):
        """Test creating MIDIConfig with default values."""
        config = MIDIConfig()
        
        self.assertEqual(config.samples_per_instrument, 10)
        self.assertIsNone(config.soundfont_path)
        self.assertIsNone(config.max_song_duration)
        self.assertEqual(config.velocity_range, (1, 127))
        self.assertEqual(config.pitch_range, (0, 127))
    
    def test_with_soundfont(self):
        """Test creating MIDIConfig with soundfont."""
        config = MIDIConfig(soundfont_path=self.soundfont_path)
        self.assertEqual(config.soundfont_path, self.soundfont_path)
    
    def test_validation_samples_per_instrument(self):
        """Test validation of samples per instrument."""
        with self.assertRaises(ValueError) as cm:
            MIDIConfig(samples_per_instrument=0)
        self.assertIn("Samples per instrument must be positive", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            MIDIConfig(samples_per_instrument=-5)
        self.assertIn("Samples per instrument must be positive", str(cm.exception))
    
    def test_validation_soundfont_path(self):
        """Test validation of soundfont path."""
        nonexistent_path = self.temp_dir / "nonexistent.sf2"
        
        with self.assertRaises(FileNotFoundError) as cm:
            MIDIConfig(soundfont_path=nonexistent_path)
        self.assertIn("Soundfont file not found", str(cm.exception))
    
    def test_validation_max_song_duration(self):
        """Test validation of max song duration."""
        with self.assertRaises(ValueError) as cm:
            MIDIConfig(max_song_duration=0)
        self.assertIn("Max song duration must be positive", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            MIDIConfig(max_song_duration=-10.0)
        self.assertIn("Max song duration must be positive", str(cm.exception))


class TestRenderConfig(unittest.TestCase):
    """Test RenderConfig class."""
    
    def test_default_creation(self):
        """Test creating RenderConfig with default values."""
        config = RenderConfig()
        
        self.assertEqual(config.harmonic_decay, 0.5)
        self.assertEqual(config.num_harmonics, 4)
        self.assertEqual(config.dominant_harmonic, 0)
        self.assertEqual(config.resonance_quality, 45.0)
        self.assertEqual(config.attack, 0.03)
        self.assertEqual(config.decay, 0.1)
        self.assertEqual(config.sustain, 0.7)
        self.assertEqual(config.release, 0.1)
    
    def test_validation_harmonic_decay(self):
        """Test validation of harmonic decay range."""
        with self.assertRaises(ValueError) as cm:
            RenderConfig(harmonic_decay=-0.1)
        self.assertIn("Harmonic decay must be between 0 and 1", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            RenderConfig(harmonic_decay=1.1)
        self.assertIn("Harmonic decay must be between 0 and 1", str(cm.exception))
    
    def test_validation_num_harmonics(self):
        """Test validation of number of harmonics."""
        with self.assertRaises(ValueError) as cm:
            RenderConfig(num_harmonics=0)
        self.assertIn("Number of harmonics must be positive", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            RenderConfig(num_harmonics=-1)
        self.assertIn("Number of harmonics must be positive", str(cm.exception))
    
    def test_validation_dominant_harmonic(self):
        """Test validation of dominant harmonic range."""
        with self.assertRaises(ValueError) as cm:
            RenderConfig(num_harmonics=4, dominant_harmonic=5)
        self.assertIn("Dominant harmonic must be between 0 and 4", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            RenderConfig(num_harmonics=4, dominant_harmonic=-1)
        self.assertIn("Dominant harmonic must be between 0 and 4", str(cm.exception))
    
    def test_validation_resonance_quality(self):
        """Test validation of resonance quality."""
        with self.assertRaises(ValueError) as cm:
            RenderConfig(resonance_quality=0)
        self.assertIn("Resonance quality must be positive", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            RenderConfig(resonance_quality=-10.0)
        self.assertIn("Resonance quality must be positive", str(cm.exception))
    
    def test_validation_adsr_envelope(self):
        """Test validation of ADSR envelope."""
        # Test ADSR sum >= 1
        with self.assertRaises(ValueError) as cm:
            RenderConfig(attack=0.5, decay=0.3, release=0.3)  # sum = 1.1
        self.assertIn("ADSR envelope (attack+decay+release) must be between 0 and 1", str(cm.exception))
        
        # Test ADSR sum = 0
        with self.assertRaises(ValueError) as cm:
            RenderConfig(attack=0, decay=0, release=0)
        self.assertIn("ADSR envelope (attack+decay+release) must be between 0 and 1", str(cm.exception))


class TestClusteringConfig(unittest.TestCase):
    """Test ClusteringConfig class."""
    
    def test_default_creation(self):
        """Test creating ClusteringConfig with default values."""
        config = ClusteringConfig()
        
        self.assertEqual(config.algorithm, "kmedoids")
        self.assertEqual(config.random_state, 42)
        self.assertEqual(config.max_iter, 300)
        self.assertEqual(config.tol, 1e-4)
    
    def test_validation_algorithm(self):
        """Test validation of clustering algorithm."""
        config = ClusteringConfig(algorithm="invalid_algorithm")
        
        with self.assertRaises(ValueError) as cm:
            config.validate()
        self.assertIn("Unsupported clustering algorithm", str(cm.exception))
    
    def test_validation_max_iter(self):
        """Test validation of max iterations."""
        config = ClusteringConfig(max_iter=0)
        
        with self.assertRaises(ValueError) as cm:
            config.validate()
        self.assertIn("Max iterations must be positive", str(cm.exception))
    
    def test_validation_tolerance(self):
        """Test validation of tolerance."""
        config = ClusteringConfig(tol=0)
        
        with self.assertRaises(ValueError) as cm:
            config.validate()
        self.assertIn("Tolerance must be positive", str(cm.exception))


class TestPathConfig(unittest.TestCase):
    """Test PathConfig class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = TestFixtures.create_temp_dir()
    
    def tearDown(self):
        """Clean up test fixtures."""
        TestFixtures.cleanup_temp_dir(self.temp_dir)
    
    def test_default_creation(self):
        """Test creating PathConfig with default values."""
        with patch('pathlib.Path.mkdir'):  # Mock mkdir to avoid actual directory creation
            config = PathConfig()
            
            self.assertEqual(config.output_dir, Path("output"))
            self.assertEqual(config.cache_dir, Path("cache"))
            self.assertEqual(config.meta_dir, Path("meta"))
            self.assertEqual(config.temp_dir, Path("temp"))
    
    def test_string_to_path_conversion(self):
        """Test conversion of string paths to Path objects."""
        with patch('pathlib.Path.mkdir'):
            config = PathConfig(
                output_dir="custom_output",
                cache_dir="custom_cache"
            )
            
            self.assertIsInstance(config.output_dir, Path)
            self.assertIsInstance(config.cache_dir, Path)
            self.assertEqual(str(config.output_dir), "custom_output")
            self.assertEqual(str(config.cache_dir), "custom_cache")


class TestAmbientMIDIConfig(unittest.TestCase):
    """Test main AmbientMIDIConfig class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = TestFixtures.create_temp_dir()
    
    def tearDown(self):
        """Clean up test fixtures."""
        TestFixtures.cleanup_temp_dir(self.temp_dir)
    
    @patch('ambientmidi.config.LoggingConfig.configure_logging')
    def test_default_creation(self, mock_configure_logging):
        """Test creating AmbientMIDIConfig with default values."""
        config = AmbientMIDIConfig()
        
        self.assertIsInstance(config.audio, AudioConfig)
        self.assertIsInstance(config.spectrogram, SpectrogramConfig)
        self.assertIsInstance(config.midi, MIDIConfig)
        self.assertIsInstance(config.render, RenderConfig)
        self.assertIsInstance(config.clustering, ClusteringConfig)
        self.assertIsInstance(config.paths, PathConfig)
        self.assertIsInstance(config.logging, LoggingConfig)
        
        # Verify logging was configured
        mock_configure_logging.assert_called_once()
    
    @patch('ambientmidi.config.LoggingConfig.configure_logging')
    def test_from_dict(self, mock_configure_logging):
        """Test creating config from dictionary."""
        config_dict = {
            "audio": {
                "sample_rate": 44100,
                "record_duration": 30.0
            },
            "midi": {
                "samples_per_instrument": 5
            }
        }
        
        config = AmbientMIDIConfig.from_dict(config_dict)
        
        self.assertEqual(config.audio.sample_rate, 44100)
        self.assertEqual(config.audio.record_duration, 30.0)
        self.assertEqual(config.midi.samples_per_instrument, 5)
    
    @patch('ambientmidi.config.LoggingConfig.configure_logging')
    def test_to_dict(self, mock_configure_logging):
        """Test converting config to dictionary."""
        config = AmbientMIDIConfig()
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertIn("audio", config_dict)
        self.assertIn("spectrogram", config_dict)
        self.assertIn("midi", config_dict)
        self.assertIn("render", config_dict)
        self.assertIn("clustering", config_dict)
        self.assertIn("paths", config_dict)
        self.assertIn("logging", config_dict)
    
    @patch('ambientmidi.config.LoggingConfig.configure_logging')
    def test_save_and_load(self, mock_configure_logging):
        """Test saving and loading configuration."""
        config = AmbientMIDIConfig()
        config.audio.sample_rate = 44100
        config.midi.samples_per_instrument = 5
        
        config_path = self.temp_dir / "test_config.json"
        
        # Save config
        config.save(config_path)
        self.assertTrue(config_path.exists())
        
        # Load config
        loaded_config = AmbientMIDIConfig.from_file(config_path)
        
        self.assertEqual(loaded_config.audio.sample_rate, 44100)
        self.assertEqual(loaded_config.midi.samples_per_instrument, 5)
    
    @patch('ambientmidi.config.LoggingConfig.configure_logging')
    def test_file_not_found(self, mock_configure_logging):
        """Test loading from non-existent file."""
        nonexistent_path = self.temp_dir / "nonexistent.json"
        
        with self.assertRaises(FileNotFoundError):
            AmbientMIDIConfig.from_file(nonexistent_path)
    
    @patch('ambientmidi.config.LoggingConfig.configure_logging')
    def test_get_cache_path(self, mock_configure_logging):
        """Test cache path generation."""
        config = AmbientMIDIConfig()
        midi_path = Path("test_song.mid")
        
        cache_path = config.get_cache_path(midi_path)
        
        self.assertEqual(cache_path.name, "test_song.json")
        self.assertTrue(str(cache_path).endswith("cache/test_song.json"))
    
    @patch('ambientmidi.config.LoggingConfig.configure_logging')
    def test_get_output_path(self, mock_configure_logging):
        """Test output path generation."""
        config = AmbientMIDIConfig()
        midi_path = Path("test_song.mid")
        
        # Without suffix
        output_path = config.get_output_path(midi_path)
        self.assertEqual(output_path.name, "test_song.wav")
        
        # With suffix
        output_path_with_suffix = config.get_output_path(midi_path, "processed")
        self.assertEqual(output_path_with_suffix.name, "test_song_processed.wav")


class TestConfigUtils(unittest.TestCase):
    """Test configuration utility functions."""
    
    @patch('ambientmidi.config.LoggingConfig.configure_logging')
    def test_get_default_config(self, mock_configure_logging):
        """Test get_default_config function."""
        config = get_default_config()
        
        self.assertIsInstance(config, AmbientMIDIConfig)
        self.assertEqual(config.audio.sample_rate, 16000)
    
    @patch('ambientmidi.config.LoggingConfig.configure_logging')
    def test_load_config_default(self, mock_configure_logging):
        """Test load_config with no path (returns default)."""
        config = load_config()
        
        self.assertIsInstance(config, AmbientMIDIConfig)
        self.assertEqual(config.audio.sample_rate, 16000)
    
    @patch('ambientmidi.config.LoggingConfig.configure_logging')
    def test_load_config_with_path(self, mock_configure_logging):
        """Test load_config with specific path."""
        temp_dir = TestFixtures.create_temp_dir()
        
        try:
            # Create a test config file
            config_path = temp_dir / "test_config.json"
            config_dict = {
                "audio": {"sample_rate": 44100},
                "midi": {"samples_per_instrument": 5}
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f)
            
            # Load the config
            config = load_config(config_path)
            
            self.assertEqual(config.audio.sample_rate, 44100)
            self.assertEqual(config.midi.samples_per_instrument, 5)
            
        finally:
            TestFixtures.cleanup_temp_dir(temp_dir)


if __name__ == '__main__':
    unittest.main()