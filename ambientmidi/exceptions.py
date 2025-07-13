"""
Exception classes for AmbientMIDI.

This module defines custom exception classes for different types of errors
that can occur during MIDI processing, audio synthesis, and rendering.
"""

from typing import Optional, Any
from pathlib import Path


class AmbientMIDIError(Exception):
    """Base exception for all AmbientMIDI errors."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ConfigurationError(AmbientMIDIError):
    """Raised when there's an error in configuration."""
    pass


class AudioProcessingError(AmbientMIDIError):
    """Raised when audio processing fails."""
    pass


class MIDIProcessingError(AmbientMIDIError):
    """Raised when MIDI processing fails."""
    pass


class RenderingError(AmbientMIDIError):
    """Raised when audio rendering fails."""
    pass


class ClusteringError(AmbientMIDIError):
    """Raised when clustering fails."""
    pass


class FeatureExtractionError(AmbientMIDIError):
    """Raised when feature extraction fails."""
    pass


class FileNotFoundError(AmbientMIDIError):
    """Raised when a required file is not found."""
    
    def __init__(self, file_path: Path, message: Optional[str] = None):
        self.file_path = file_path
        if message is None:
            message = f"File not found: {file_path}"
        super().__init__(message, {"file_path": str(file_path)})


class InvalidInputError(AmbientMIDIError):
    """Raised when input validation fails."""
    pass


class ProcessingTimeoutError(AmbientMIDIError):
    """Raised when processing takes too long."""
    pass


class AudioRecordingError(AudioProcessingError):
    """Raised when audio recording fails."""
    pass


class SoundFontError(MIDIProcessingError):
    """Raised when there's an issue with the soundfont."""
    pass


class OnsetDetectionError(AudioProcessingError):
    """Raised when onset detection fails."""
    pass


class SpectrogramError(AudioProcessingError):
    """Raised when spectrogram computation fails."""
    pass


def handle_error(error: Exception, context: str = "") -> AmbientMIDIError:
    """
    Convert generic exceptions to AmbientMIDI exceptions.
    
    Args:
        error: The original exception
        context: Additional context about where the error occurred
        
    Returns:
        An appropriate AmbientMIDIError subclass
    """
    if isinstance(error, AmbientMIDIError):
        return error
    
    error_type = type(error).__name__
    message = f"{context}: {error_type}: {str(error)}" if context else f"{error_type}: {str(error)}"
    
    # Map common Python exceptions to AmbientMIDI exceptions
    if isinstance(error, ValueError):
        return InvalidInputError(message)
    elif isinstance(error, FileNotFoundError):
        return FileNotFoundError(Path(str(error)), message)
    elif isinstance(error, TimeoutError):
        return ProcessingTimeoutError(message)
    elif isinstance(error, ImportError):
        return ConfigurationError(message)
    else:
        return AmbientMIDIError(message)


class ErrorHandler:
    """Context manager for handling errors in a consistent way."""
    
    def __init__(self, context: str, reraise: bool = True):
        self.context = context
        self.reraise = reraise
        self.error: Optional[Exception] = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error = handle_error(exc_val, self.context)
            if self.reraise:
                raise self.error
            return True  # Suppress the exception
        return False
    
    def has_error(self) -> bool:
        """Check if an error occurred."""
        return self.error is not None