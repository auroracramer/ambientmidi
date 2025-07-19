"""
Unit tests for the AmbientMIDI audio processing module.

Tests cover:
- Audio recording and loading
- Audio processing functions (rescaling, normalization, etc.)
- Error handling and validation
- Mock external dependencies for isolated testing
"""

import unittest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from ambientmidi.audio import (
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
from ambientmidi.config import AudioConfig
from ambientmidi.exceptions import (
    AudioProcessingError,
    AudioRecordingError,
    FileNotFoundError,
    InvalidInputError
)
from tests import TestFixtures


class TestAudioRecording(unittest.TestCase):
    """Test audio recording functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fixtures = TestFixtures()
        self.mock_audio_data = self.fixtures.create_mock_audio()
    
    @patch('ambientmidi.audio.pyaudio.PyAudio')
    @patch('ambientmidi.audio.nr.reduce_noise')
    def test_record_audio_success(self, mock_noise_reduce, mock_pyaudio):
        """Test successful audio recording."""
        # Mock PyAudio
        mock_pa = MagicMock()
        mock_pyaudio.return_value = mock_pa
        
        mock_device_info = {'name': 'Test Device', 'index': 0}
        mock_pa.get_default_input_device_info.return_value = mock_device_info
        
        mock_stream = MagicMock()
        mock_pa.open.return_value = mock_stream
        
        # Create mock audio data (1 second at 16kHz)
        sample_rate = 16000
        duration = 1.0
        samples = int(sample_rate * duration)
        chunk_size = 1024
        
        # Create chunks of audio data
        audio_chunks = []
        remaining_samples = samples
        while remaining_samples > 0:
            chunk_samples = min(chunk_size, remaining_samples)
            chunk_data = np.random.random(chunk_samples).astype(np.float32).tobytes()
            audio_chunks.append(chunk_data)
            remaining_samples -= chunk_samples
        
        mock_stream.read.side_effect = audio_chunks
        mock_noise_reduce.return_value = self.mock_audio_data
        
        # Record audio
        result = record_audio(duration=duration, sample_rate=sample_rate, denoise=True)
        
        # Verify calls
        mock_pa.open.assert_called_once()
        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()
        mock_pa.terminate.assert_called_once()
        mock_noise_reduce.assert_called_once()
        
        # Verify result
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(self.mock_audio_data))
    
    def test_record_audio_invalid_duration(self):
        """Test recording with invalid duration."""
        with self.assertRaises(InvalidInputError) as cm:
            record_audio(duration=0)
        self.assertIn("Duration must be positive", str(cm.exception))
        
        with self.assertRaises(InvalidInputError) as cm:
            record_audio(duration=-5.0)
        self.assertIn("Duration must be positive", str(cm.exception))
    
    def test_record_audio_invalid_sample_rate(self):
        """Test recording with invalid sample rate."""
        with self.assertRaises(InvalidInputError) as cm:
            record_audio(sample_rate=0)
        self.assertIn("Sample rate must be positive", str(cm.exception))
        
        with self.assertRaises(InvalidInputError) as cm:
            record_audio(sample_rate=-1000)
        self.assertIn("Sample rate must be positive", str(cm.exception))
    
    def test_record_audio_invalid_channels(self):
        """Test recording with invalid number of channels."""
        with self.assertRaises(InvalidInputError) as cm:
            record_audio(channels=0)
        self.assertIn("Channels must be 1 or 2", str(cm.exception))
        
        with self.assertRaises(InvalidInputError) as cm:
            record_audio(channels=3)
        self.assertIn("Channels must be 1 or 2", str(cm.exception))
    
    @patch('ambientmidi.audio.pyaudio.PyAudio')
    def test_record_audio_pyaudio_error(self, mock_pyaudio):
        """Test audio recording with PyAudio error."""
        mock_pyaudio.side_effect = Exception("PyAudio error")
        
        with self.assertRaises(AudioRecordingError) as cm:
            record_audio()
        self.assertIn("Failed to record audio", str(cm.exception))


class TestAudioLoading(unittest.TestCase):
    """Test audio loading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fixtures = TestFixtures()
        self.temp_dir = self.fixtures.create_temp_dir()
        self.mock_audio_data = self.fixtures.create_mock_audio()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.fixtures.cleanup_temp_dir(self.temp_dir)
    
    @patch('ambientmidi.audio.sf.read')
    @patch('ambientmidi.audio.resampy.resample')
    def test_load_audio_success(self, mock_resample, mock_sf_read):
        """Test successful audio loading."""
        # Create test file
        audio_file = self.temp_dir / "test.wav"
        audio_file.touch()
        
        # Mock soundfile read
        original_sr = 44100
        target_sr = 16000
        mock_sf_read.return_value = (self.mock_audio_data.reshape(-1, 1), original_sr)
        mock_resample.return_value = self.mock_audio_data
        
        # Load audio
        result = load_audio(audio_file, target_sr)
        
        # Verify calls
        mock_sf_read.assert_called_once_with(str(audio_file), always_2d=True)
        mock_resample.assert_called_once_with(self.mock_audio_data, original_sr, target_sr)
        
        # Verify result
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(self.mock_audio_data))
    
    def test_load_audio_file_not_found(self):
        """Test loading non-existent audio file."""
        nonexistent_file = self.temp_dir / "nonexistent.wav"
        
        with self.assertRaises(FileNotFoundError):
            load_audio(nonexistent_file, 16000)
    
    @patch('ambientmidi.audio.sf.read')
    def test_load_audio_stereo_to_mono(self, mock_sf_read):
        """Test loading stereo audio and converting to mono."""
        # Create test file
        audio_file = self.temp_dir / "test.wav"
        audio_file.touch()
        
        # Create stereo audio (2 channels)
        stereo_audio = np.random.random((1000, 2)).astype(np.float32)
        mock_sf_read.return_value = (stereo_audio, 16000)
        
        # Load audio
        result = load_audio(audio_file, 16000)
        
        # Verify mono conversion
        self.assertEqual(result.ndim, 1)
        self.assertEqual(len(result), 1000)
    
    @patch('ambientmidi.audio.sf.read')
    def test_load_audio_with_target_duration(self, mock_sf_read):
        """Test loading audio with target duration."""
        # Create test file
        audio_file = self.temp_dir / "test.wav"
        audio_file.touch()
        
        # Mock audio data (2 seconds)
        sample_rate = 16000
        audio_data = np.random.random((2 * sample_rate, 1)).astype(np.float32)
        mock_sf_read.return_value = (audio_data, sample_rate)
        
        # Load with target duration of 1 second (should trim)
        result = load_audio(audio_file, sample_rate, target_duration=1.0)
        
        self.assertEqual(len(result), sample_rate)  # 1 second worth
        
        # Load with target duration of 3 seconds (should pad)
        result_padded = load_audio(audio_file, sample_rate, target_duration=3.0)
        
        self.assertEqual(len(result_padded), 3 * sample_rate)  # 3 seconds worth
    
    @patch('ambientmidi.audio.sf.read')
    def test_load_audio_sf_error(self, mock_sf_read):
        """Test audio loading with soundfile error."""
        # Create test file
        audio_file = self.temp_dir / "test.wav"
        audio_file.touch()
        
        mock_sf_read.side_effect = Exception("Soundfile error")
        
        with self.assertRaises(AudioProcessingError) as cm:
            load_audio(audio_file, 16000)
        self.assertIn("Failed to load audio", str(cm.exception))


class TestAudioProcessing(unittest.TestCase):
    """Test audio processing functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fixtures = TestFixtures()
        self.mock_audio_data = self.fixtures.create_mock_audio()
    
    def test_rescale_audio_float32(self):
        """Test rescaling float32 audio."""
        # Create test audio in range [-1, 1]
        audio = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        
        result = rescale_audio(audio)
        
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(len(result), len(audio))
        # Result should be scaled to [-2^31, 2^31] range
        self.assertGreaterEqual(result.min(), -2**31)
        self.assertLessEqual(result.max(), 2**31)
    
    def test_rescale_audio_int16(self):
        """Test rescaling int16 audio."""
        # Create test audio in int16 range
        audio = np.array([-32768, -16384, 0, 16384, 32767], dtype=np.int16)
        
        result = rescale_audio(audio)
        
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(len(result), len(audio))
    
    def test_rescale_audio_empty(self):
        """Test rescaling empty audio array."""
        audio = np.array([], dtype=np.float32)
        
        with self.assertRaises(InvalidInputError) as cm:
            rescale_audio(audio)
        self.assertIn("Cannot rescale empty audio array", str(cm.exception))
    
    def test_rescale_audio_invalid_dtype(self):
        """Test rescaling audio with invalid dtype."""
        audio = np.array([1, 2, 3], dtype=np.complex64)
        
        with self.assertRaises(AudioProcessingError) as cm:
            rescale_audio(audio)
        self.assertIn("Invalid audio dtype", str(cm.exception))
    
    @patch('ambientmidi.audio.ln.normalize.peak')
    @patch('ambientmidi.audio.ln.Meter')
    def test_normalize_loudness_peak(self, mock_meter, mock_peak):
        """Test loudness normalization with peak mode."""
        mock_peak.return_value = self.mock_audio_data
        
        result = normalize_loudness(self.mock_audio_data, 16000, use_peak=True)
        
        mock_peak.assert_called_once_with(self.mock_audio_data, -14.0)
        self.assertIsInstance(result, np.ndarray)
    
    @patch('ambientmidi.audio.ln.normalize.loudness')
    @patch('ambientmidi.audio.ln.Meter')
    def test_normalize_loudness_integrated(self, mock_meter, mock_loudness):
        """Test loudness normalization with integrated mode."""
        mock_meter_instance = MagicMock()
        mock_meter.return_value = mock_meter_instance
        mock_meter_instance.integrated_loudness.return_value = -20.0
        mock_loudness.return_value = self.mock_audio_data
        
        result = normalize_loudness(self.mock_audio_data, 16000, use_peak=False)
        
        mock_loudness.assert_called_once_with(self.mock_audio_data, -20.0, -14.0)
        self.assertIsInstance(result, np.ndarray)
    
    def test_normalize_loudness_empty_audio(self):
        """Test normalization with empty audio."""
        audio = np.array([], dtype=np.float32)
        
        with self.assertRaises(InvalidInputError) as cm:
            normalize_loudness(audio, 16000)
        self.assertIn("Cannot normalize empty audio array", str(cm.exception))
    
    def test_normalize_loudness_invalid_sample_rate(self):
        """Test normalization with invalid sample rate."""
        with self.assertRaises(InvalidInputError) as cm:
            normalize_loudness(self.mock_audio_data, 0)
        self.assertIn("Sample rate must be positive", str(cm.exception))
    
    def test_apply_fade(self):
        """Test applying fade-in and fade-out."""
        audio = np.ones(1600, dtype=np.float32)  # 0.1 seconds at 16kHz
        sample_rate = 16000
        
        result = apply_fade(
            audio, 
            fade_in_duration=0.05,  # 50ms
            fade_out_duration=0.05,  # 50ms
            sample_rate=sample_rate
        )
        
        self.assertEqual(len(result), len(audio))
        # Check that fade-in starts from 0
        self.assertAlmostEqual(result[0], 0.0, places=5)
        # Check that fade-out ends at 0
        self.assertAlmostEqual(result[-1], 0.0, places=5)
        # Check that middle is unchanged
        middle_idx = len(result) // 2
        self.assertAlmostEqual(result[middle_idx], 1.0, places=5)
    
    def test_apply_fade_empty_audio(self):
        """Test applying fade to empty audio."""
        audio = np.array([], dtype=np.float32)
        
        result = apply_fade(audio)
        
        self.assertEqual(len(result), 0)
    
    def test_detect_silence(self):
        """Test silence detection."""
        # Create audio with silence in the middle
        sample_rate = 16000
        duration = 1.0
        samples = int(sample_rate * duration)
        
        audio = np.ones(samples, dtype=np.float32) * 0.5  # Loud audio
        # Insert silence in the middle
        silence_start = samples // 3
        silence_end = 2 * samples // 3
        audio[silence_start:silence_end] = 0.005  # Below threshold
        
        silence_mask, silence_regions = detect_silence(
            audio, 
            sample_rate, 
            threshold=0.01, 
            min_duration=0.1
        )
        
        self.assertEqual(len(silence_mask), len(audio))
        self.assertTrue(np.any(silence_mask))  # Should detect some silence
        self.assertGreater(len(silence_regions), 0)  # Should find silence regions
    
    def test_detect_silence_empty_audio(self):
        """Test silence detection with empty audio."""
        audio = np.array([], dtype=np.float32)
        
        silence_mask, silence_regions = detect_silence(audio, 16000)
        
        self.assertEqual(len(silence_mask), 0)
        self.assertEqual(len(silence_regions), 0)
    
    def test_validate_audio_array_valid(self):
        """Test validation of valid audio array."""
        # Should not raise any exception
        validate_audio_array(self.mock_audio_data)
        validate_audio_array(self.mock_audio_data, "test context")
    
    def test_validate_audio_array_empty(self):
        """Test validation of empty audio array."""
        audio = np.array([], dtype=np.float32)
        
        with self.assertRaises(InvalidInputError) as cm:
            validate_audio_array(audio)
        self.assertIn("Empty audio array", str(cm.exception))
        
        with self.assertRaises(InvalidInputError) as cm:
            validate_audio_array(audio, "test context")
        self.assertIn("Empty audio array in test context", str(cm.exception))
    
    def test_validate_audio_array_multidimensional(self):
        """Test validation of multi-dimensional audio array."""
        audio = np.random.random((100, 2)).astype(np.float32)
        
        with self.assertRaises(InvalidInputError) as cm:
            validate_audio_array(audio)
        self.assertIn("Audio must be 1D, got 2D", str(cm.exception))
    
    def test_validate_audio_array_non_finite(self):
        """Test validation of audio array with non-finite values."""
        audio = np.array([1.0, 2.0, np.inf, 3.0], dtype=np.float32)
        
        with self.assertRaises(InvalidInputError) as cm:
            validate_audio_array(audio)
        self.assertIn("Audio contains non-finite values", str(cm.exception))


class TestAudioConvenienceFunctions(unittest.TestCase):
    """Test convenience functions for audio operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fixtures = TestFixtures()
        self.temp_dir = self.fixtures.create_temp_dir()
        self.config = AudioConfig()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.fixtures.cleanup_temp_dir(self.temp_dir)
    
    @patch('ambientmidi.audio.load_audio')
    @patch('ambientmidi.audio.nr.reduce_noise')
    @patch('ambientmidi.audio.normalize_loudness')
    def test_load_and_prepare_audio(self, mock_normalize, mock_noise_reduce, mock_load):
        """Test load and prepare audio convenience function."""
        # Create test file
        audio_file = self.temp_dir / "test.wav"
        audio_file.touch()
        
        mock_audio = self.fixtures.create_mock_audio()
        mock_load.return_value = mock_audio
        mock_noise_reduce.return_value = mock_audio
        mock_normalize.return_value = mock_audio
        
        result = load_and_prepare_audio(audio_file, self.config)
        
        # Verify function calls
        mock_load.assert_called_once_with(audio_file, self.config.sample_rate)
        mock_noise_reduce.assert_called_once()
        mock_normalize.assert_called_once()
        
        self.assertIsInstance(result, np.ndarray)
    
    @patch('ambientmidi.audio.record_audio')
    @patch('ambientmidi.audio.normalize_loudness')
    def test_record_and_prepare_audio(self, mock_normalize, mock_record):
        """Test record and prepare audio convenience function."""
        mock_audio = self.fixtures.create_mock_audio()
        mock_record.return_value = mock_audio
        mock_normalize.return_value = mock_audio
        
        result = record_and_prepare_audio(self.config)
        
        # Verify function calls
        mock_record.assert_called_once_with(
            duration=self.config.record_duration,
            sample_rate=self.config.sample_rate,
            denoise=self.config.denoise_enabled
        )
        mock_normalize.assert_called_once()
        
        self.assertIsInstance(result, np.ndarray)
    
    @patch('ambientmidi.audio.load_audio')
    @patch('ambientmidi.audio.nr.reduce_noise')
    def test_load_and_prepare_audio_no_denoise(self, mock_noise_reduce, mock_load):
        """Test load and prepare audio without denoising."""
        # Create test file
        audio_file = self.temp_dir / "test.wav"
        audio_file.touch()
        
        # Disable denoising
        self.config.denoise_enabled = False
        
        mock_audio = self.fixtures.create_mock_audio()
        mock_load.return_value = mock_audio
        
        result = load_and_prepare_audio(audio_file, self.config)
        
        # Verify denoising was not called
        mock_noise_reduce.assert_not_called()
        
        self.assertIsInstance(result, np.ndarray)


if __name__ == '__main__':
    unittest.main()