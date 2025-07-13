"""
Processing pipeline for AmbientMIDI.

This module defines the main processing pipeline that orchestrates
MIDI processing, audio synthesis, and rendering in a modular way.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, Union
from pathlib import Path
from dataclasses import dataclass
import json
import time

from .config import AmbientMIDIConfig
from .exceptions import (
    AmbientMIDIError, 
    MIDIProcessingError, 
    AudioProcessingError, 
    RenderingError,
    FileNotFoundError,
    ErrorHandler
)


logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of a processing step."""
    data: Any
    metadata: Dict[str, Any]
    processing_time: float
    success: bool
    error: Optional[AmbientMIDIError] = None


class ProcessingStep(ABC):
    """Abstract base class for processing steps."""
    
    def __init__(self, name: str, config: AmbientMIDIConfig):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def process(self, input_data: Any, **kwargs) -> ProcessingResult:
        """Process the input data and return a result."""
        pass
    
    def validate_input(self, input_data: Any) -> None:
        """Validate input data. Override in subclasses."""
        pass
    
    def cleanup(self) -> None:
        """Clean up resources. Override in subclasses."""
        pass


class MIDIPreprocessingStep(ProcessingStep):
    """MIDI preprocessing step."""
    
    def __init__(self, config: AmbientMIDIConfig):
        super().__init__("midi_preprocessing", config)
    
    def process(self, midi_path: Path, **kwargs) -> ProcessingResult:
        """Process MIDI file and extract note events."""
        start_time = time.time()
        
        try:
            self.validate_input(midi_path)
            
            # Check cache first
            cache_path = self.config.get_cache_path(midi_path)
            if cache_path.exists():
                self.logger.info(f"Loading cached MIDI data from {cache_path}")
                with open(cache_path, 'r') as f:
                    midi_info = json.load(f)
            else:
                self.logger.info(f"Processing MIDI file: {midi_path}")
                
                # Import here to avoid circular imports
                from .midi import preprocess_midi_file
                
                midi_info = preprocess_midi_file(
                    str(midi_path),
                    sample_rate=self.config.audio.sample_rate,
                    samples_per_instr=self.config.midi.samples_per_instrument,
                    soundfont_path=self.config.midi.soundfont_path,
                    max_song_duration=self.config.midi.max_song_duration
                )
                
                # Cache the result
                with open(cache_path, 'w') as f:
                    from .utils import NpEncoder
                    json.dump(midi_info, f, cls=NpEncoder)
                
                self.logger.info(f"Cached MIDI data to {cache_path}")
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                data=midi_info,
                metadata={"source_file": str(midi_path), "cached": cache_path.exists()},
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            with ErrorHandler("MIDI preprocessing", reraise=False) as handler:
                raise e
            
            return ProcessingResult(
                data=None,
                metadata={"source_file": str(midi_path)},
                processing_time=time.time() - start_time,
                success=False,
                error=handler.error
            )
    
    def validate_input(self, midi_path: Path) -> None:
        """Validate MIDI file path."""
        if not midi_path.exists():
            raise FileNotFoundError(midi_path)
        
        if not midi_path.suffix.lower() in ['.mid', '.midi']:
            raise MIDIProcessingError(
                f"Invalid MIDI file extension: {midi_path.suffix}",
                {"file_path": str(midi_path)}
            )


class AudioAcquisitionStep(ProcessingStep):
    """Audio acquisition step (recording or loading)."""
    
    def __init__(self, config: AmbientMIDIConfig):
        super().__init__("audio_acquisition", config)
    
    def process(self, input_recording_path: Optional[Path] = None, **kwargs) -> ProcessingResult:
        """Acquire audio data through recording or loading."""
        start_time = time.time()
        
        try:
            from .audio import load_audio, record_audio
            
            if input_recording_path:
                self.logger.info(f"Loading audio from: {input_recording_path}")
                self.validate_input(input_recording_path)
                audio = load_audio(input_recording_path, self.config.audio.sample_rate)
                metadata = {"source": "file", "file_path": str(input_recording_path)}
            else:
                self.logger.info(f"Recording audio for {self.config.audio.record_duration} seconds")
                audio = record_audio(
                    duration=self.config.audio.record_duration,
                    sample_rate=self.config.audio.sample_rate,
                    denoise=self.config.audio.denoise_enabled
                )
                metadata = {"source": "recording", "duration": self.config.audio.record_duration}
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                data=audio,
                metadata=metadata,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            with ErrorHandler("Audio acquisition", reraise=False) as handler:
                raise e
            
            return ProcessingResult(
                data=None,
                metadata={"source": "file" if input_recording_path else "recording"},
                processing_time=time.time() - start_time,
                success=False,
                error=handler.error
            )
    
    def validate_input(self, audio_path: Path) -> None:
        """Validate audio file path."""
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)


class EventProcessingStep(ProcessingStep):
    """Event processing step (onset detection and clustering)."""
    
    def __init__(self, config: AmbientMIDIConfig):
        super().__init__("event_processing", config)
    
    def process(self, audio_data: Any, **kwargs) -> ProcessingResult:
        """Process audio to detect events and cluster them."""
        start_time = time.time()
        
        try:
            from .events import compute_pcengram, get_onsets, get_event_clip_dicts
            from .cluster import get_clip_clusters
            
            self.logger.info("Computing PCEN spectrogram")
            pcengram = compute_pcengram(audio_data, sample_rate=self.config.audio.sample_rate)
            
            self.logger.info("Detecting onsets")
            onset_idxs, onset_frames, onset_env = get_onsets(
                pcengram, 
                sample_rate=self.config.audio.sample_rate
            )
            
            self.logger.info("Extracting event clips")
            event_clip_list = get_event_clip_dicts(
                audio_data, 
                onset_idxs, 
                sample_rate=self.config.audio.sample_rate
            )
            
            self.logger.info("Clustering events")
            env_clusters_to_events = get_clip_clusters(event_clip_list)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                data=env_clusters_to_events,
                metadata={
                    "num_onsets": len(onset_idxs),
                    "num_events": len(event_clip_list),
                    "num_clusters": len(env_clusters_to_events)
                },
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            with ErrorHandler("Event processing", reraise=False) as handler:
                raise e
            
            return ProcessingResult(
                data=None,
                metadata={},
                processing_time=time.time() - start_time,
                success=False,
                error=handler.error
            )


class AudioRenderingStep(ProcessingStep):
    """Audio rendering step."""
    
    def __init__(self, config: AmbientMIDIConfig):
        super().__init__("audio_rendering", config)
    
    def process(self, midi_info: Dict, event_clusters: Any, **kwargs) -> ProcessingResult:
        """Render final audio output."""
        start_time = time.time()
        
        try:
            from .render import render_song_from_events
            
            self.logger.info("Rendering audio from events")
            audio_output = render_song_from_events(
                midi_info,
                event_clusters,
                max_song_duration=self.config.midi.max_song_duration,
                sample_rate=self.config.audio.sample_rate
            )
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                data=audio_output,
                metadata={
                    "duration": len(audio_output) / self.config.audio.sample_rate,
                    "sample_rate": self.config.audio.sample_rate
                },
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            with ErrorHandler("Audio rendering", reraise=False) as handler:
                raise e
            
            return ProcessingResult(
                data=None,
                metadata={},
                processing_time=time.time() - start_time,
                success=False,
                error=handler.error
            )


class AmbientMIDIPipeline:
    """Main processing pipeline for AmbientMIDI."""
    
    def __init__(self, config: AmbientMIDIConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.Pipeline")
        
        # Initialize processing steps
        self.steps = {
            "midi_preprocessing": MIDIPreprocessingStep(config),
            "audio_acquisition": AudioAcquisitionStep(config),
            "event_processing": EventProcessingStep(config),
            "audio_rendering": AudioRenderingStep(config)
        }
        
        self.results: Dict[str, ProcessingResult] = {}
        
    def process(
        self, 
        midi_path: Path, 
        output_path: Path, 
        input_recording_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, ProcessingResult]:
        """
        Run the complete processing pipeline.
        
        Args:
            midi_path: Path to MIDI file
            output_path: Path for output audio file
            input_recording_path: Optional path to input recording
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary of processing results for each step
        """
        self.logger.info(f"Starting AmbientMIDI pipeline")
        self.logger.info(f"MIDI file: {midi_path}")
        self.logger.info(f"Output file: {output_path}")
        
        try:
            # Step 1: MIDI preprocessing
            self._update_progress(progress_callback, "midi_preprocessing", 0.0)
            midi_result = self.steps["midi_preprocessing"].process(midi_path)
            self.results["midi_preprocessing"] = midi_result
            
            if not midi_result.success:
                self.logger.error(f"MIDI preprocessing failed: {midi_result.error}")
                return self.results
            
            # Step 2: Audio acquisition
            self._update_progress(progress_callback, "audio_acquisition", 0.25)
            audio_result = self.steps["audio_acquisition"].process(input_recording_path)
            self.results["audio_acquisition"] = audio_result
            
            if not audio_result.success:
                self.logger.error(f"Audio acquisition failed: {audio_result.error}")
                return self.results
            
            # Step 3: Event processing
            self._update_progress(progress_callback, "event_processing", 0.5)
            event_result = self.steps["event_processing"].process(audio_result.data)
            self.results["event_processing"] = event_result
            
            if not event_result.success:
                self.logger.error(f"Event processing failed: {event_result.error}")
                return self.results
            
            # Step 4: Audio rendering
            self._update_progress(progress_callback, "audio_rendering", 0.75)
            render_result = self.steps["audio_rendering"].process(
                midi_result.data, 
                event_result.data
            )
            self.results["audio_rendering"] = render_result
            
            if not render_result.success:
                self.logger.error(f"Audio rendering failed: {render_result.error}")
                return self.results
            
            # Save output
            self._update_progress(progress_callback, "saving", 0.9)
            self._save_output(render_result.data, output_path)
            
            self._update_progress(progress_callback, "complete", 1.0)
            self.logger.info("Pipeline completed successfully")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            self._cleanup()
    
    def _update_progress(
        self, 
        callback: Optional[Callable[[str, float], None]], 
        step: str, 
        progress: float
    ) -> None:
        """Update progress if callback is provided."""
        if callback:
            callback(step, progress)
    
    def _save_output(self, audio_data: Any, output_path: Path) -> None:
        """Save audio output to file."""
        try:
            import soundfile as sf
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), audio_data, self.config.audio.sample_rate)
            
            self.logger.info(f"Saved output to: {output_path}")
            
        except Exception as e:
            raise RenderingError(f"Failed to save output: {e}")
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        for step in self.steps.values():
            try:
                step.cleanup()
            except Exception as e:
                self.logger.warning(f"Error during cleanup: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the pipeline execution."""
        total_time = sum(r.processing_time for r in self.results.values())
        
        return {
            "total_processing_time": total_time,
            "steps": {
                name: {
                    "success": result.success,
                    "processing_time": result.processing_time,
                    "metadata": result.metadata,
                    "error": str(result.error) if result.error else None
                }
                for name, result in self.results.items()
            }
        }
    
    def save_summary(self, summary_path: Path) -> None:
        """Save pipeline summary to file."""
        summary = self.get_summary()
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Saved pipeline summary to: {summary_path}")


def create_pipeline(config: Optional[AmbientMIDIConfig] = None) -> AmbientMIDIPipeline:
    """Create a new processing pipeline."""
    if config is None:
        from .config import get_default_config
        config = get_default_config()
    
    return AmbientMIDIPipeline(config)