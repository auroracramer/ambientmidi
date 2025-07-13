"""
Audio processing module for AmbientMIDI.

This module provides functions for audio recording, loading, and processing
with robust error handling and configuration support.
"""

import logging
import pyaudio
import resampy
import soundfile as sf
import numpy as np
import noisereduce as nr
import pyloudnorm as ln
from pathlib import Path
from typing import Optional, Union, Tuple

from .config import AudioConfig
from .exceptions import (
    AudioProcessingError, 
    AudioRecordingError, 
    FileNotFoundError,
    ErrorHandler,
    InvalidInputError
)


logger = logging.getLogger(__name__)


def record_audio(
    duration: float = 60.0, 
    sample_rate: int = 16000, 
    denoise: bool = True,
    channels: int = 1,
    chunk_size: int = 1024
) -> np.ndarray:
    """
    Record audio from the default input device.
    
    Args:
        duration: Duration in seconds to record
        sample_rate: Sample rate for recording
        denoise: Whether to apply noise reduction
        channels: Number of audio channels (1 for mono, 2 for stereo)
        chunk_size: Size of audio chunks for streaming
        
    Returns:
        Recorded audio as numpy array
        
    Raises:
        AudioRecordingError: If recording fails
        InvalidInputError: If parameters are invalid
    """
    with ErrorHandler("Audio recording"):
        logger.info(f"Recording audio for {duration} seconds at {sample_rate} Hz")
        
        # Validate parameters
        if duration <= 0:
            raise InvalidInputError(f"Duration must be positive, got {duration}")
        if sample_rate <= 0:
            raise InvalidInputError(f"Sample rate must be positive, got {sample_rate}")
        if channels not in [1, 2]:
            raise InvalidInputError(f"Channels must be 1 or 2, got {channels}")
        
        p = pyaudio.PyAudio()
        
        try:
            # Get default input device info
            default_device = p.get_default_input_device_info()
            logger.debug(f"Using input device: {default_device['name']}")
            
            # Open audio stream
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk_size,
                input_device_index=default_device['index']
            )
            
            # Calculate total frames needed
            total_frames = int(duration * sample_rate)
            frames_recorded = 0
            audio_data = []
            
            logger.debug("Starting audio recording...")
            
            # Record in chunks
            while frames_recorded < total_frames:
                remaining_frames = total_frames - frames_recorded
                chunk_frames = min(chunk_size, remaining_frames)
                
                data = stream.read(chunk_frames)
                audio_data.append(data)
                frames_recorded += chunk_frames
                
                # Log progress periodically
                if frames_recorded % (sample_rate * 5) == 0:  # Every 5 seconds
                    progress = frames_recorded / total_frames
                    logger.debug(f"Recording progress: {progress:.1%}")
            
            # Stop and close stream
            stream.stop_stream()
            stream.close()
            
            # Convert to numpy array
            audio_bytes = b''.join(audio_data)
            audio = np.frombuffer(audio_bytes, dtype=np.float32)
            
            # Handle stereo to mono conversion if needed
            if channels == 2:
                audio = audio.reshape(-1, 2)
                audio = audio.mean(axis=1)  # Convert to mono
            
            logger.info(f"Successfully recorded {len(audio) / sample_rate:.2f}s of audio")
            
            # Apply processing
            audio = rescale_audio(audio)
            
            if denoise:
                logger.info("Applying noise reduction...")
                audio = nr.reduce_noise(y=audio, sr=sample_rate)
            
            return audio
            
        except Exception as e:
            raise AudioRecordingError(f"Failed to record audio: {e}")
        finally:
            p.terminate()


def load_audio(
    path: Union[str, Path], 
    sample_rate: int,
    normalize: bool = True,
    target_duration: Optional[float] = None
) -> np.ndarray:
    """
    Load audio file and resample if necessary.
    
    Args:
        path: Path to audio file
        sample_rate: Target sample rate
        normalize: Whether to normalize audio
        target_duration: Optional target duration in seconds
        
    Returns:
        Loaded audio as numpy array
        
    Raises:
        FileNotFoundError: If file doesn't exist
        AudioProcessingError: If loading fails
    """
    with ErrorHandler("Audio loading"):
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(path)
        
        logger.info(f"Loading audio from: {path}")
        
        try:
            # Load audio file
            audio, original_sr = sf.read(str(path), always_2d=True)
            
            # Convert to mono if stereo
            if audio.shape[1] > 1:
                audio = audio.mean(axis=1)  # Average channels
                logger.debug("Converted stereo to mono")
            else:
                audio = audio.squeeze()
            
            logger.debug(f"Original sample rate: {original_sr}, target: {sample_rate}")
            
            # Resample if necessary
            if original_sr != sample_rate:
                logger.info(f"Resampling from {original_sr} Hz to {sample_rate} Hz")
                audio = resampy.resample(audio, original_sr, sample_rate)
            
            # Trim or pad to target duration if specified
            if target_duration is not None:
                target_samples = int(target_duration * sample_rate)
                if len(audio) > target_samples:
                    audio = audio[:target_samples]
                    logger.debug(f"Trimmed audio to {target_duration}s")
                elif len(audio) < target_samples:
                    padding = target_samples - len(audio)
                    audio = np.pad(audio, (0, padding), mode='constant')
                    logger.debug(f"Padded audio to {target_duration}s")
            
            # Normalize if requested
            if normalize:
                audio = rescale_audio(audio)
            
            logger.info(f"Successfully loaded {len(audio) / sample_rate:.2f}s of audio")
            
            return audio
            
        except Exception as e:
            raise AudioProcessingError(f"Failed to load audio from {path}: {e}")


def rescale_audio(audio: np.ndarray) -> np.ndarray:
    """
    Rescale audio to standardized float32 format.
    
    Args:
        audio: Input audio array
        
    Returns:
        Rescaled audio array
        
    Raises:
        AudioProcessingError: If rescaling fails
    """
    with ErrorHandler("Audio rescaling"):
        if audio.size == 0:
            raise InvalidInputError("Cannot rescale empty audio array")
        
        # Handle different input types
        if audio.dtype.kind == 'i':
            # Integer input
            max_val = max(np.iinfo(audio.dtype).max, -np.iinfo(audio.dtype).min)
            audio_scaled = audio.astype('float64') / max_val
        elif audio.dtype.kind == 'f':
            # Float input
            audio_scaled = audio.astype('float64')
        else:
            raise AudioProcessingError(f'Invalid audio dtype: {audio.dtype}')
        
        # Clip to valid range
        audio_scaled = np.clip(audio_scaled, -1.0, 1.0)
        
        # Map to the range [-2**31, 2**31] for compatibility
        return (audio_scaled * (2**31)).astype('float32')


def normalize_loudness(
    audio: np.ndarray, 
    sample_rate: int,
    target_db_lufs: float = -14.0, 
    use_peak: bool = True
) -> np.ndarray:
    """
    Normalize audio loudness to target level.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate of audio
        target_db_lufs: Target loudness in LUFS
        use_peak: Whether to use peak normalization or integrated loudness
        
    Returns:
        Normalized audio array
        
    Raises:
        AudioProcessingError: If normalization fails
    """
    with ErrorHandler("Audio normalization"):
        if audio.size == 0:
            raise InvalidInputError("Cannot normalize empty audio array")
        
        if sample_rate <= 0:
            raise InvalidInputError(f"Sample rate must be positive, got {sample_rate}")
        
        logger.debug(f"Normalizing audio to {target_db_lufs} LUFS")
        
        try:
            if use_peak:
                normalized = ln.normalize.peak(audio, target_db_lufs)
            else:
                meter = ln.Meter(sample_rate)
                loudness = meter.integrated_loudness(audio)
                normalized = ln.normalize.loudness(audio, loudness, target_db_lufs)
            
            logger.debug("Audio normalization completed")
            return normalized
            
        except Exception as e:
            raise AudioProcessingError(f"Failed to normalize audio: {e}")


def apply_fade(
    audio: np.ndarray,
    fade_in_duration: float = 0.1,
    fade_out_duration: float = 0.1,
    sample_rate: int = 16000
) -> np.ndarray:
    """
    Apply fade-in and fade-out to audio.
    
    Args:
        audio: Input audio array
        fade_in_duration: Fade-in duration in seconds
        fade_out_duration: Fade-out duration in seconds
        sample_rate: Sample rate of audio
        
    Returns:
        Audio with fades applied
        
    Raises:
        AudioProcessingError: If fade application fails
    """
    with ErrorHandler("Audio fade application"):
        if audio.size == 0:
            return audio
        
        audio_out = audio.copy()
        
        # Apply fade-in
        if fade_in_duration > 0:
            fade_in_samples = int(fade_in_duration * sample_rate)
            fade_in_samples = min(fade_in_samples, len(audio_out))
            
            fade_in_curve = np.linspace(0, 1, fade_in_samples)
            audio_out[:fade_in_samples] *= fade_in_curve
        
        # Apply fade-out
        if fade_out_duration > 0:
            fade_out_samples = int(fade_out_duration * sample_rate)
            fade_out_samples = min(fade_out_samples, len(audio_out))
            
            fade_out_curve = np.linspace(1, 0, fade_out_samples)
            audio_out[-fade_out_samples:] *= fade_out_curve
        
        return audio_out


def detect_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold: float = 0.01,
    min_duration: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect silence regions in audio.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate of audio
        threshold: Silence threshold (0-1)
        min_duration: Minimum duration for silence regions in seconds
        
    Returns:
        Tuple of (silence_mask, silence_regions)
        
    Raises:
        AudioProcessingError: If silence detection fails
    """
    with ErrorHandler("Silence detection"):
        if audio.size == 0:
            return np.array([]), np.array([])
        
        # Calculate audio energy
        energy = np.abs(audio)
        
        # Apply threshold
        silence_mask = energy < threshold
        
        # Find silence regions
        silence_changes = np.diff(silence_mask.astype(int))
        silence_starts = np.where(silence_changes == 1)[0] + 1
        silence_ends = np.where(silence_changes == -1)[0] + 1
        
        # Handle edge cases
        if silence_mask[0]:
            silence_starts = np.concatenate([[0], silence_starts])
        if silence_mask[-1]:
            silence_ends = np.concatenate([silence_ends, [len(audio)]])
        
        # Filter by minimum duration
        min_samples = int(min_duration * sample_rate)
        silence_regions = []
        
        for start, end in zip(silence_starts, silence_ends):
            if end - start >= min_samples:
                silence_regions.append((start, end))
        
        return silence_mask, np.array(silence_regions)


def validate_audio_array(audio: np.ndarray, context: str = "") -> None:
    """
    Validate audio array for common issues.
    
    Args:
        audio: Audio array to validate
        context: Context information for error messages
        
    Raises:
        InvalidInputError: If validation fails
    """
    if audio.size == 0:
        raise InvalidInputError(f"Empty audio array{' in ' + context if context else ''}")
    
    if audio.ndim != 1:
        raise InvalidInputError(f"Audio must be 1D, got {audio.ndim}D{' in ' + context if context else ''}")
    
    if not np.isfinite(audio).all():
        raise InvalidInputError(f"Audio contains non-finite values{' in ' + context if context else ''}")
    
    if audio.dtype not in [np.float32, np.float64]:
        logger.warning(f"Audio dtype is {audio.dtype}, expected float32 or float64{' in ' + context if context else ''}")


# Convenience functions for common audio operations
def load_and_prepare_audio(
    path: Union[str, Path],
    config: AudioConfig
) -> np.ndarray:
    """
    Load and prepare audio according to configuration.
    
    Args:
        path: Path to audio file
        config: Audio configuration
        
    Returns:
        Prepared audio array
    """
    audio = load_audio(path, config.sample_rate)
    
    if config.denoise_enabled:
        audio = nr.reduce_noise(y=audio, sr=config.sample_rate)
    
    if config.use_peak_normalization:
        audio = normalize_loudness(
            audio, 
            config.sample_rate, 
            config.target_db_lufs, 
            use_peak=True
        )
    
    return audio


def record_and_prepare_audio(config: AudioConfig) -> np.ndarray:
    """
    Record and prepare audio according to configuration.
    
    Args:
        config: Audio configuration
        
    Returns:
        Prepared audio array
    """
    audio = record_audio(
        duration=config.record_duration,
        sample_rate=config.sample_rate,
        denoise=config.denoise_enabled
    )
    
    if config.use_peak_normalization:
        audio = normalize_loudness(
            audio, 
            config.sample_rate, 
            config.target_db_lufs, 
            use_peak=True
        )
    
    return audio