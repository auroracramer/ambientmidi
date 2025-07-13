"""
Event detection and processing module for AmbientMIDI.

This module provides functions for onset detection, PCEN spectrogram computation,
and event clip extraction with robust error handling and configuration support.
"""

import logging
import librosa
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from .config import AudioConfig, SpectrogramConfig
from .exceptions import (
    OnsetDetectionError,
    SpectrogramError,
    AudioProcessingError,
    ErrorHandler,
    InvalidInputError
)
from .audio import normalize_loudness, validate_audio_array
from .features import get_feature_dict
from .utils import qtile


logger = logging.getLogger(__name__)


def compute_pcengram(
    audio: np.ndarray,
    sample_rate: int = 16000,
    window_size_ms: float = 25.0,
    hop_size_ms: float = 10.0,
    n_mels: int = 40,
    power: float = 2.0
) -> np.ndarray:
    """
    Compute PCEN (Per-Channel Energy Normalization) spectrogram.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate of the audio
        window_size_ms: Window size in milliseconds
        hop_size_ms: Hop size in milliseconds
        n_mels: Number of mel bands
        power: Power for spectrogram computation
        
    Returns:
        PCEN spectrogram
        
    Raises:
        SpectrogramError: If PCEN computation fails
        InvalidInputError: If input validation fails
    """
    with ErrorHandler("PCEN spectrogram computation"):
        # Validate input
        validate_audio_array(audio, "PCEN computation")
        
        if sample_rate <= 0:
            raise InvalidInputError(f"Sample rate must be positive, got {sample_rate}")
        if window_size_ms <= 0 or hop_size_ms <= 0:
            raise InvalidInputError("Window and hop sizes must be positive")
        if n_mels <= 0:
            raise InvalidInputError(f"Number of mel bands must be positive, got {n_mels}")
        if power <= 0:
            raise InvalidInputError(f"Power must be positive, got {power}")
        
        logger.debug(f"Computing PCEN spectrogram with {n_mels} mel bands")
        
        try:
            # Calculate parameters
            n_fft = int((window_size_ms / 1000) * sample_rate)
            hop_length = int((hop_size_ms / 1000) * sample_rate)
            
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                power=power
            )
            
            # Apply PCEN normalization
            pcen_spec = librosa.pcen(mel_spec)
            
            logger.debug(f"PCEN spectrogram shape: {pcen_spec.shape}")
            
            return pcen_spec
            
        except Exception as e:
            raise SpectrogramError(f"Failed to compute PCEN spectrogram: {e}")


def get_onsets(
    pcengram: np.ndarray,
    sample_rate: int = 16000,
    hop_size_ms: float = 10.0,
    onset_threshold: float = 0.5,
    min_onset_separation: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect onsets from PCEN spectrogram.
    
    Args:
        pcengram: PCEN spectrogram
        sample_rate: Sample rate of the audio
        hop_size_ms: Hop size in milliseconds
        onset_threshold: Threshold for onset detection
        min_onset_separation: Minimum separation between onsets in seconds
        
    Returns:
        Tuple of (onset_indices, onset_frames, onset_envelope)
        
    Raises:
        OnsetDetectionError: If onset detection fails
        InvalidInputError: If input validation fails
    """
    with ErrorHandler("Onset detection"):
        # Validate input
        if pcengram.size == 0:
            raise InvalidInputError("Cannot detect onsets from empty spectrogram")
        
        if pcengram.ndim != 2:
            raise InvalidInputError(f"PCEN spectrogram must be 2D, got {pcengram.ndim}D")
        
        if sample_rate <= 0:
            raise InvalidInputError(f"Sample rate must be positive, got {sample_rate}")
        
        if hop_size_ms <= 0:
            raise InvalidInputError(f"Hop size must be positive, got {hop_size_ms}")
        
        if onset_threshold < 0 or onset_threshold > 1:
            raise InvalidInputError(f"Onset threshold must be between 0 and 1, got {onset_threshold}")
        
        logger.debug(f"Detecting onsets with threshold {onset_threshold}")
        
        try:
            hop_length = int((hop_size_ms / 1000) * sample_rate)
            
            # Compute onset strength
            onset_envelope = librosa.onset.onset_strength(
                S=pcengram,
                sr=sample_rate,
                hop_length=hop_length,
                aggregate=np.median
            )
            
            # Detect onset frames
            onset_frames = librosa.onset.onset_detect(
                onset_envelope=onset_envelope,
                sr=sample_rate,
                hop_length=hop_length,
                threshold=onset_threshold,
                pre_max=int(0.03 * sample_rate / hop_length),  # 30ms pre-max
                post_max=int(0.03 * sample_rate / hop_length),  # 30ms post-max
                pre_avg=int(0.1 * sample_rate / hop_length),   # 100ms pre-avg
                post_avg=int(0.1 * sample_rate / hop_length),  # 100ms post-avg
                delta=onset_threshold
            )
            
            # Convert frames to sample indices
            onset_indices = librosa.frames_to_samples(onset_frames, hop_length=hop_length)
            
            # Filter by minimum separation
            if min_onset_separation > 0:
                min_separation_samples = int(min_onset_separation * sample_rate)
                filtered_indices = []
                
                for i, onset_idx in enumerate(onset_indices):
                    if i == 0 or onset_idx - filtered_indices[-1] >= min_separation_samples:
                        filtered_indices.append(onset_idx)
                
                onset_indices = np.array(filtered_indices)
                # Update frames accordingly
                onset_frames = librosa.samples_to_frames(onset_indices, hop_length=hop_length)
            
            logger.info(f"Detected {len(onset_indices)} onsets")
            
            return onset_indices, onset_frames, onset_envelope
            
        except Exception as e:
            raise OnsetDetectionError(f"Failed to detect onsets: {e}")


def truncate_silence(
    audio: np.ndarray,
    n_fft: int,
    hop_length: int,
    min_clip_length: int,
    silence_threshold: float = 0.01
) -> np.ndarray:
    """
    Truncate silence from audio clip.
    
    Args:
        audio: Input audio array
        n_fft: FFT size
        hop_length: Hop length
        min_clip_length: Minimum clip length in samples
        silence_threshold: Silence threshold
        
    Returns:
        Truncated audio array
        
    Raises:
        AudioProcessingError: If truncation fails
    """
    with ErrorHandler("Silence truncation"):
        validate_audio_array(audio, "silence truncation")
        
        if len(audio) < min_clip_length:
            return audio
        
        try:
            # Compute short-time energy
            energy = librosa.feature.rms(
                y=audio,
                frame_length=n_fft,
                hop_length=hop_length
            )[0]
            
            # Find non-silent regions
            non_silent_frames = energy > silence_threshold
            
            if not non_silent_frames.any():
                # All frames are silent, return minimal clip
                return audio[:min_clip_length]
            
            # Find start and end of non-silent region
            start_frame = np.where(non_silent_frames)[0][0]
            end_frame = np.where(non_silent_frames)[0][-1] + 1
            
            # Convert to samples
            start_sample = start_frame * hop_length
            end_sample = min(end_frame * hop_length, len(audio))
            
            # Ensure minimum length
            if end_sample - start_sample < min_clip_length:
                if start_sample > 0:
                    start_sample = max(0, end_sample - min_clip_length)
                else:
                    end_sample = min(len(audio), start_sample + min_clip_length)
            
            return audio[start_sample:end_sample]
            
        except Exception as e:
            raise AudioProcessingError(f"Failed to truncate silence: {e}")


def get_event_clip_dicts(
    audio: np.ndarray,
    onset_idx_list: List[int],
    sample_rate: int = 16000,
    min_clip_size_s: float = 0.125,
    max_clip_size_s: float = 1.0,
    truncate_silence: bool = False,
    window_size_ms: float = 25.0,
    hop_size_ms: float = 10.0
) -> List[Dict[str, Any]]:
    """
    Extract event clips from audio at onset locations.
    
    Args:
        audio: Input audio array
        onset_idx_list: List of onset indices
        sample_rate: Sample rate of the audio
        min_clip_size_s: Minimum clip size in seconds
        max_clip_size_s: Maximum clip size in seconds
        truncate_silence: Whether to truncate silence
        window_size_ms: Window size in milliseconds
        hop_size_ms: Hop size in milliseconds
        
    Returns:
        List of event clip dictionaries
        
    Raises:
        AudioProcessingError: If clip extraction fails
    """
    with ErrorHandler("Event clip extraction"):
        validate_audio_array(audio, "event clip extraction")
        
        if not onset_idx_list:
            logger.warning("No onsets provided for clip extraction")
            return []
        
        if sample_rate <= 0:
            raise InvalidInputError(f"Sample rate must be positive, got {sample_rate}")
        
        if min_clip_size_s <= 0 or max_clip_size_s <= 0:
            raise InvalidInputError("Clip sizes must be positive")
        
        if min_clip_size_s >= max_clip_size_s:
            raise InvalidInputError("Min clip size must be less than max clip size")
        
        logger.debug(f"Extracting clips from {len(onset_idx_list)} onsets")
        
        try:
            # Calculate clip parameters
            min_clip_length = int(min_clip_size_s * sample_rate)
            max_clip_length = int(max_clip_size_s * sample_rate)
            n_fft = int((window_size_ms / 1000) * sample_rate)
            hop_length = int((hop_size_ms / 1000) * sample_rate)
            
            event_clips = []
            
            for i, onset_idx in enumerate(onset_idx_list):
                try:
                    # Determine clip boundaries
                    start_idx = max(0, onset_idx - min_clip_length // 4)
                    
                    if i < len(onset_idx_list) - 1:
                        # Not the last onset, use next onset as boundary
                        next_onset = onset_idx_list[i + 1]
                        end_idx = min(start_idx + max_clip_length, next_onset)
                    else:
                        # Last onset, use max clip length
                        end_idx = min(start_idx + max_clip_length, len(audio))
                    
                    # Ensure minimum length
                    if end_idx - start_idx < min_clip_length:
                        end_idx = min(start_idx + min_clip_length, len(audio))
                    
                    # Extract clip
                    clip = audio[start_idx:end_idx]
                    
                    # Truncate silence if requested
                    if truncate_silence:
                        clip = truncate_silence(
                            clip, n_fft, hop_length, min_clip_length
                        )
                    
                    # Ensure clip is not empty
                    if len(clip) == 0:
                        logger.warning(f"Empty clip at onset {i}, skipping")
                        continue
                    
                    # Compute audio features
                    features = get_feature_dict(
                        clip,
                        sample_rate,
                        features=("mfcc", "pitch_hz", "tonality", "spectral_centroid", "zero_crossing_rate")
                    )
                    
                    # Create event dictionary
                    event_dict = {
                        "audio_clip": clip,
                        "onset_idx": onset_idx,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "duration": len(clip) / sample_rate,
                        "clip_idx": i,
                        **features
                    }
                    
                    event_clips.append(event_dict)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract clip at onset {i}: {e}")
                    continue
            
            logger.info(f"Successfully extracted {len(event_clips)} event clips")
            
            return event_clips
            
        except Exception as e:
            raise AudioProcessingError(f"Failed to extract event clips: {e}")


def filter_events_by_energy(
    event_clips: List[Dict[str, Any]],
    energy_threshold: float = 0.01,
    energy_percentile: float = 25.0
) -> List[Dict[str, Any]]:
    """
    Filter event clips by energy level.
    
    Args:
        event_clips: List of event clip dictionaries
        energy_threshold: Absolute energy threshold
        energy_percentile: Percentile threshold for relative filtering
        
    Returns:
        Filtered list of event clips
        
    Raises:
        AudioProcessingError: If filtering fails
    """
    with ErrorHandler("Event energy filtering"):
        if not event_clips:
            return event_clips
        
        try:
            # Calculate energy for each clip
            energies = []
            for event in event_clips:
                clip = event["audio_clip"]
                energy = np.mean(np.abs(clip))
                energies.append(energy)
            
            energies = np.array(energies)
            
            # Apply absolute threshold
            energy_mask = energies > energy_threshold
            
            # Apply percentile threshold
            if energy_percentile > 0:
                energy_threshold_percentile = np.percentile(energies, energy_percentile)
                energy_mask = energy_mask & (energies > energy_threshold_percentile)
            
            # Filter events
            filtered_events = [
                event for event, keep in zip(event_clips, energy_mask) if keep
            ]
            
            logger.info(f"Filtered {len(event_clips)} events to {len(filtered_events)} by energy")
            
            return filtered_events
            
        except Exception as e:
            raise AudioProcessingError(f"Failed to filter events by energy: {e}")


def compute_onset_strength_envelope(
    audio: np.ndarray,
    sample_rate: int = 16000,
    hop_size_ms: float = 10.0,
    aggregate_function: str = "median"
) -> np.ndarray:
    """
    Compute onset strength envelope from audio.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate of the audio
        hop_size_ms: Hop size in milliseconds
        aggregate_function: Aggregation function for onset strength
        
    Returns:
        Onset strength envelope
        
    Raises:
        OnsetDetectionError: If computation fails
    """
    with ErrorHandler("Onset strength computation"):
        validate_audio_array(audio, "onset strength computation")
        
        if sample_rate <= 0:
            raise InvalidInputError(f"Sample rate must be positive, got {sample_rate}")
        
        if hop_size_ms <= 0:
            raise InvalidInputError(f"Hop size must be positive, got {hop_size_ms}")
        
        # Map aggregate function names
        agg_functions = {
            "median": np.median,
            "mean": np.mean,
            "max": np.max,
            "min": np.min
        }
        
        if aggregate_function not in agg_functions:
            raise InvalidInputError(f"Invalid aggregate function: {aggregate_function}")
        
        try:
            hop_length = int((hop_size_ms / 1000) * sample_rate)
            
            onset_envelope = librosa.onset.onset_strength(
                y=audio,
                sr=sample_rate,
                hop_length=hop_length,
                aggregate=agg_functions[aggregate_function]
            )
            
            return onset_envelope
            
        except Exception as e:
            raise OnsetDetectionError(f"Failed to compute onset strength: {e}")


# Configuration-based convenience functions
def compute_pcengram_from_config(
    audio: np.ndarray,
    audio_config: AudioConfig,
    spec_config: SpectrogramConfig
) -> np.ndarray:
    """
    Compute PCEN spectrogram using configuration objects.
    
    Args:
        audio: Input audio array
        audio_config: Audio configuration
        spec_config: Spectrogram configuration
        
    Returns:
        PCEN spectrogram
    """
    return compute_pcengram(
        audio=audio,
        sample_rate=audio_config.sample_rate,
        window_size_ms=spec_config.window_size_ms,
        hop_size_ms=spec_config.hop_size_ms,
        n_mels=spec_config.n_mels,
        power=spec_config.power
    )


def get_onsets_from_config(
    pcengram: np.ndarray,
    audio_config: AudioConfig,
    spec_config: SpectrogramConfig,
    onset_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect onsets using configuration objects.
    
    Args:
        pcengram: PCEN spectrogram
        audio_config: Audio configuration
        spec_config: Spectrogram configuration
        onset_threshold: Threshold for onset detection
        
    Returns:
        Tuple of (onset_indices, onset_frames, onset_envelope)
    """
    return get_onsets(
        pcengram=pcengram,
        sample_rate=audio_config.sample_rate,
        hop_size_ms=spec_config.hop_size_ms,
        onset_threshold=onset_threshold
    )


def get_event_clips_from_config(
    audio: np.ndarray,
    onset_idx_list: List[int],
    audio_config: AudioConfig,
    spec_config: SpectrogramConfig
) -> List[Dict[str, Any]]:
    """
    Extract event clips using configuration objects.
    
    Args:
        audio: Input audio array
        onset_idx_list: List of onset indices
        audio_config: Audio configuration
        spec_config: Spectrogram configuration
        
    Returns:
        List of event clip dictionaries
    """
    return get_event_clip_dicts(
        audio=audio,
        onset_idx_list=onset_idx_list,
        sample_rate=audio_config.sample_rate,
        min_clip_size_s=audio_config.min_clip_size_s,
        max_clip_size_s=audio_config.max_clip_size_s,
        window_size_ms=spec_config.window_size_ms,
        hop_size_ms=spec_config.hop_size_ms
    )