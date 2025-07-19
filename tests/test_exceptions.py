"""
Unit tests for the AmbientMIDI exceptions module.

Tests cover:
- Exception hierarchy and inheritance
- Error context and details
- Error handler functionality
- Exception conversion and mapping
- Context manager behavior
"""

import unittest
from pathlib import Path
from unittest.mock import patch

from ambientmidi.exceptions import (
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
    handle_error,
    ErrorHandler
)


class TestExceptionHierarchy(unittest.TestCase):
    """Test exception class hierarchy and inheritance."""
    
    def test_base_exception_inheritance(self):
        """Test that all custom exceptions inherit from AmbientMIDIError."""
        exception_classes = [
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
            SpectrogramError
        ]
        
        for exc_class in exception_classes:
            self.assertTrue(issubclass(exc_class, AmbientMIDIError))
    
    def test_specialized_inheritance(self):
        """Test that specialized exceptions inherit from appropriate base classes."""
        # Audio-related exceptions should inherit from AudioProcessingError
        audio_exceptions = [AudioRecordingError, OnsetDetectionError, SpectrogramError]
        for exc_class in audio_exceptions:
            self.assertTrue(issubclass(exc_class, AudioProcessingError))
        
        # MIDI-related exceptions should inherit from MIDIProcessingError
        midi_exceptions = [SoundFontError]
        for exc_class in midi_exceptions:
            self.assertTrue(issubclass(exc_class, MIDIProcessingError))


class TestAmbientMIDIError(unittest.TestCase):
    """Test base AmbientMIDIError class."""
    
    def test_basic_creation(self):
        """Test creating basic AmbientMIDIError."""
        error = AmbientMIDIError("Test error message")
        
        self.assertEqual(str(error), "Test error message")
        self.assertEqual(error.message, "Test error message")
        self.assertEqual(error.details, {})
    
    def test_creation_with_details(self):
        """Test creating AmbientMIDIError with details."""
        details = {"file_path": "/test/path", "line_number": 42}
        error = AmbientMIDIError("Test error with details", details)
        
        self.assertEqual(error.message, "Test error with details")
        self.assertEqual(error.details, details)
    
    def test_string_representation_with_details(self):
        """Test string representation includes details."""
        details = {"file_path": "/test/path", "error_code": 123}
        error = AmbientMIDIError("Test error", details)
        
        error_str = str(error)
        self.assertIn("Test error", error_str)
        self.assertIn("file_path=/test/path", error_str)
        self.assertIn("error_code=123", error_str)
    
    def test_string_representation_without_details(self):
        """Test string representation without details."""
        error = AmbientMIDIError("Test error")
        
        self.assertEqual(str(error), "Test error")


class TestSpecializedExceptions(unittest.TestCase):
    """Test specialized exception classes."""
    
    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Invalid configuration")
        
        self.assertIsInstance(error, AmbientMIDIError)
        self.assertEqual(str(error), "Invalid configuration")
    
    def test_audio_processing_error(self):
        """Test AudioProcessingError."""
        error = AudioProcessingError("Audio processing failed")
        
        self.assertIsInstance(error, AmbientMIDIError)
        self.assertEqual(str(error), "Audio processing failed")
    
    def test_midi_processing_error(self):
        """Test MIDIProcessingError."""
        error = MIDIProcessingError("MIDI parsing failed")
        
        self.assertIsInstance(error, AmbientMIDIError)
        self.assertEqual(str(error), "MIDI parsing failed")
    
    def test_rendering_error(self):
        """Test RenderingError."""
        error = RenderingError("Audio rendering failed")
        
        self.assertIsInstance(error, AmbientMIDIError)
        self.assertEqual(str(error), "Audio rendering failed")
    
    def test_clustering_error(self):
        """Test ClusteringError."""
        error = ClusteringError("Clustering failed")
        
        self.assertIsInstance(error, AmbientMIDIError)
        self.assertEqual(str(error), "Clustering failed")
    
    def test_feature_extraction_error(self):
        """Test FeatureExtractionError."""
        error = FeatureExtractionError("Feature extraction failed")
        
        self.assertIsInstance(error, AmbientMIDIError)
        self.assertEqual(str(error), "Feature extraction failed")
    
    def test_invalid_input_error(self):
        """Test InvalidInputError."""
        error = InvalidInputError("Invalid input provided")
        
        self.assertIsInstance(error, AmbientMIDIError)
        self.assertEqual(str(error), "Invalid input provided")
    
    def test_processing_timeout_error(self):
        """Test ProcessingTimeoutError."""
        error = ProcessingTimeoutError("Processing timed out")
        
        self.assertIsInstance(error, AmbientMIDIError)
        self.assertEqual(str(error), "Processing timed out")


class TestFileNotFoundError(unittest.TestCase):
    """Test custom FileNotFoundError class."""
    
    def test_creation_with_path(self):
        """Test creating FileNotFoundError with Path object."""
        file_path = Path("/test/nonexistent.txt")
        error = FileNotFoundError(file_path)
        
        self.assertIsInstance(error, AmbientMIDIError)
        self.assertEqual(error.file_path, file_path)
        self.assertIn("File not found: /test/nonexistent.txt", str(error))
        self.assertEqual(error.details["file_path"], str(file_path))
    
    def test_creation_with_custom_message(self):
        """Test creating FileNotFoundError with custom message."""
        file_path = Path("/test/missing.wav")
        custom_message = "Audio file is missing"
        error = FileNotFoundError(file_path, custom_message)
        
        self.assertEqual(error.file_path, file_path)
        self.assertEqual(str(error), custom_message)
        self.assertEqual(error.details["file_path"], str(file_path))


class TestAudioSubclassExceptions(unittest.TestCase):
    """Test audio processing subclass exceptions."""
    
    def test_audio_recording_error(self):
        """Test AudioRecordingError."""
        error = AudioRecordingError("Recording device not available")
        
        self.assertIsInstance(error, AudioProcessingError)
        self.assertIsInstance(error, AmbientMIDIError)
        self.assertEqual(str(error), "Recording device not available")
    
    def test_onset_detection_error(self):
        """Test OnsetDetectionError."""
        error = OnsetDetectionError("Onset detection failed")
        
        self.assertIsInstance(error, AudioProcessingError)
        self.assertIsInstance(error, AmbientMIDIError)
        self.assertEqual(str(error), "Onset detection failed")
    
    def test_spectrogram_error(self):
        """Test SpectrogramError."""
        error = SpectrogramError("Spectrogram computation failed")
        
        self.assertIsInstance(error, AudioProcessingError)
        self.assertIsInstance(error, AmbientMIDIError)
        self.assertEqual(str(error), "Spectrogram computation failed")


class TestMIDISubclassExceptions(unittest.TestCase):
    """Test MIDI processing subclass exceptions."""
    
    def test_soundfont_error(self):
        """Test SoundFontError."""
        error = SoundFontError("SoundFont file is corrupted")
        
        self.assertIsInstance(error, MIDIProcessingError)
        self.assertIsInstance(error, AmbientMIDIError)
        self.assertEqual(str(error), "SoundFont file is corrupted")


class TestHandleError(unittest.TestCase):
    """Test handle_error function."""
    
    def test_handle_ambient_midi_error(self):
        """Test handling existing AmbientMIDIError."""
        original_error = AudioProcessingError("Original error")
        
        result = handle_error(original_error)
        
        self.assertIs(result, original_error)  # Should return the same object
    
    def test_handle_value_error(self):
        """Test handling ValueError."""
        original_error = ValueError("Invalid value")
        
        result = handle_error(original_error, "test context")
        
        self.assertIsInstance(result, InvalidInputError)
        self.assertIn("test context", str(result))
        self.assertIn("ValueError", str(result))
        self.assertIn("Invalid value", str(result))
    
    def test_handle_file_not_found_error(self):
        """Test handling standard FileNotFoundError."""
        original_error = FileNotFoundError("No such file")
        
        result = handle_error(original_error)
        
        self.assertIsInstance(result, FileNotFoundError)  # Custom FileNotFoundError
        self.assertIsInstance(result, AmbientMIDIError)
    
    def test_handle_timeout_error(self):
        """Test handling TimeoutError."""
        original_error = TimeoutError("Operation timed out")
        
        result = handle_error(original_error, "processing")
        
        self.assertIsInstance(result, ProcessingTimeoutError)
        self.assertIn("processing", str(result))
        self.assertIn("TimeoutError", str(result))
    
    def test_handle_import_error(self):
        """Test handling ImportError."""
        original_error = ImportError("Module not found")
        
        result = handle_error(original_error)
        
        self.assertIsInstance(result, ConfigurationError)
        self.assertIn("ImportError", str(result))
        self.assertIn("Module not found", str(result))
    
    def test_handle_generic_error(self):
        """Test handling generic exception."""
        original_error = RuntimeError("Unexpected error")
        
        result = handle_error(original_error, "test operation")
        
        self.assertIsInstance(result, AmbientMIDIError)
        self.assertIn("test operation", str(result))
        self.assertIn("RuntimeError", str(result))
        self.assertIn("Unexpected error", str(result))
    
    def test_handle_error_without_context(self):
        """Test handling error without context."""
        original_error = ValueError("Test error")
        
        result = handle_error(original_error)
        
        self.assertIsInstance(result, InvalidInputError)
        self.assertIn("ValueError", str(result))
        self.assertIn("Test error", str(result))
        self.assertNotIn(":", str(result))  # No context prefix


class TestErrorHandler(unittest.TestCase):
    """Test ErrorHandler context manager."""
    
    def test_no_exception(self):
        """Test ErrorHandler when no exception occurs."""
        handler = ErrorHandler("test context")
        
        with handler:
            # No exception should occur
            result = 42
        
        self.assertFalse(handler.has_error())
        self.assertIsNone(handler.error)
    
    def test_exception_with_reraise(self):
        """Test ErrorHandler with reraise=True (default)."""
        handler = ErrorHandler("test context", reraise=True)
        
        with self.assertRaises(InvalidInputError) as cm:
            with handler:
                raise ValueError("Test error")
        
        self.assertTrue(handler.has_error())
        self.assertIsInstance(handler.error, InvalidInputError)
        self.assertIn("test context", str(cm.exception))
        self.assertIn("ValueError", str(cm.exception))
    
    def test_exception_without_reraise(self):
        """Test ErrorHandler with reraise=False."""
        handler = ErrorHandler("test context", reraise=False)
        
        # Should not raise exception
        with handler:
            raise ValueError("Test error")
        
        self.assertTrue(handler.has_error())
        self.assertIsInstance(handler.error, InvalidInputError)
        self.assertIn("test context", str(handler.error))
    
    def test_ambient_midi_error_handling(self):
        """Test ErrorHandler with existing AmbientMIDIError."""
        handler = ErrorHandler("test context", reraise=False)
        original_error = AudioProcessingError("Audio error")
        
        with handler:
            raise original_error
        
        self.assertTrue(handler.has_error())
        self.assertIs(handler.error, original_error)  # Should be the same object
    
    def test_context_manager_protocol(self):
        """Test that ErrorHandler properly implements context manager protocol."""
        handler = ErrorHandler("test")
        
        # Test __enter__
        enter_result = handler.__enter__()
        self.assertIs(enter_result, handler)
        
        # Test __exit__ with no exception
        exit_result = handler.__exit__(None, None, None)
        self.assertFalse(exit_result)  # Should not suppress anything
        
        # Test __exit__ with exception
        try:
            raise ValueError("Test")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
            exit_result = handler.__exit__(*exc_info)
            self.assertTrue(exit_result)  # Should suppress the exception
    
    def test_multiple_context_uses(self):
        """Test using the same ErrorHandler multiple times."""
        handler = ErrorHandler("test context", reraise=False)
        
        # First use - no error
        with handler:
            pass
        
        self.assertFalse(handler.has_error())
        
        # Second use - with error
        with handler:
            raise ValueError("Second error")
        
        self.assertTrue(handler.has_error())
        self.assertIsInstance(handler.error, InvalidInputError)
    
    def test_error_details_preservation(self):
        """Test that error details are preserved through handling."""
        handler = ErrorHandler("audio processing", reraise=False)
        
        class DetailedError(Exception):
            def __init__(self, message, error_code):
                super().__init__(message)
                self.error_code = error_code
        
        with handler:
            raise DetailedError("Complex error", 404)
        
        self.assertTrue(handler.has_error())
        # The error message should contain information about the original exception
        error_str = str(handler.error)
        self.assertIn("audio processing", error_str)
        self.assertIn("DetailedError", error_str)
        self.assertIn("Complex error", error_str)


class TestExceptionIntegration(unittest.TestCase):
    """Integration tests for exception handling."""
    
    def test_chained_error_handling(self):
        """Test handling exceptions through multiple layers."""
        def inner_function():
            raise ValueError("Inner error")
        
        def middle_function():
            try:
                inner_function()
            except Exception as e:
                raise AudioProcessingError("Middle layer error") from e
        
        def outer_function():
            with ErrorHandler("outer context", reraise=False) as handler:
                middle_function()
            return handler
        
        handler = outer_function()
        
        self.assertTrue(handler.has_error())
        self.assertIsInstance(handler.error, AudioProcessingError)
        # The original AudioProcessingError should be preserved
        self.assertEqual(str(handler.error), "Middle layer error")
    
    def test_error_context_accumulation(self):
        """Test that error context accumulates through nested handlers."""
        with ErrorHandler("outer context", reraise=False) as outer_handler:
            with ErrorHandler("inner context", reraise=False) as inner_handler:
                raise ValueError("Original error")
        
        # Inner handler should catch the error
        self.assertTrue(inner_handler.has_error())
        self.assertIn("inner context", str(inner_handler.error))
        
        # Outer handler should not catch anything since inner didn't reraise
        self.assertFalse(outer_handler.has_error())
    
    def test_error_suppression_and_reraise(self):
        """Test combining error suppression and reraising."""
        def process_with_recovery():
            with ErrorHandler("processing", reraise=False) as handler:
                raise ValueError("Processing failed")
            
            if handler.has_error():
                # Log the error and try recovery
                recovery_attempted = True
                # If recovery fails, reraise as different exception
                raise ProcessingTimeoutError("Recovery failed")
            
            return "success"
        
        with self.assertRaises(ProcessingTimeoutError):
            process_with_recovery()


if __name__ == '__main__':
    unittest.main()