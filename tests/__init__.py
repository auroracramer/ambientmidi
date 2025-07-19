"""
AmbientMIDI Test Suite

This package contains comprehensive unit tests for all components of the AmbientMIDI system.
The tests are organized by module and include:

- Unit tests for individual functions and classes
- Integration tests for component interactions
- Mocking of external dependencies for isolated testing
- Performance and stress testing
- Configuration and validation testing
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
import numpy as np

# Add the parent directory to the path for importing ambientmidi
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Test utilities and fixtures
class TestFixtures:
    """Common test fixtures and utilities."""
    
    @staticmethod
    def create_temp_dir() -> Path:
        """Create a temporary directory for test files."""
        return Path(tempfile.mkdtemp())
    
    @staticmethod
    def cleanup_temp_dir(temp_dir: Path) -> None:
        """Clean up temporary directory."""
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    @staticmethod
    def create_mock_audio(sample_rate: int = 16000, duration: float = 1.0) -> np.ndarray:
        """Create mock audio data for testing."""
        samples = int(sample_rate * duration)
        # Create a simple sine wave
        t = np.linspace(0, duration, samples, False)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        return audio
    
    @staticmethod
    def create_mock_spectrogram(n_mels: int = 40, n_frames: int = 100) -> np.ndarray:
        """Create mock spectrogram data for testing."""
        return np.random.rand(n_mels, n_frames).astype(np.float32)
    
    @staticmethod
    def create_mock_midi_info() -> dict:
        """Create mock MIDI info structure for testing."""
        return {
            'name': 'test_midi',
            'duration': 30.0,
            'instr_to_events': {
                'piano': {
                    'events': [
                        {
                            'midi_note': {
                                'program': 0,
                                'name': 'piano',
                                'is_drum': False,
                                'velocity': 80,
                                'pitch': 60,
                                'pitch_hz': 261.63,
                                'start': 0.0,
                                'end': 1.0,
                                'duration': 1.0
                            },
                            'mfcc': np.random.rand(40),
                            'pitch_hz': 261.63,
                            'tonality': 0.8
                        }
                    ],
                    'mean_mfcc': 0.5,
                    'mean_tonality': 0.8,
                    'is_drum': False,
                    'num_events': 1
                }
            }
        }

# Test constants
TEST_SAMPLE_RATE = 16000
TEST_DURATION = 1.0
TEST_N_MELS = 40
TEST_HOP_SIZE_MS = 10.0
TEST_WINDOW_SIZE_MS = 25.0

# Mock data paths
MOCK_MIDI_PATH = "test_files/mock.mid"
MOCK_AUDIO_PATH = "test_files/mock.wav"
MOCK_OUTPUT_PATH = "test_files/output.wav"