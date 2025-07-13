"""
AmbientMIDI Synthesizer

A robust, configurable MIDI synthesis system for creating ambient audio
from MIDI files using recorded or live audio input.
"""

import sys
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

from ambientmidi.config import AmbientMIDIConfig, load_config
from ambientmidi.pipeline import create_pipeline
from ambientmidi.exceptions import AmbientMIDIError, ConfigurationError


def create_argument_parser() -> ArgumentParser:
    """Create and configure the argument parser."""
    parser = ArgumentParser(
        description="AmbientMIDI Synthesizer - Create ambient audio from MIDI files",
        formatter_class=ArgumentParser.__class__
    )
    
    # Required arguments
    parser.add_argument(
        "midi_path", 
        type=Path,
        help="Path to the MIDI file to process"
    )
    
    parser.add_argument(
        "output_path", 
        type=Path,
        help="Path for the output audio file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--config", 
        type=Path,
        help="Path to configuration file (JSON format)"
    )
    
    parser.add_argument(
        "--input-recording", 
        type=Path,
        help="Path to input recording file (if not provided, will record live audio)"
    )
    
    parser.add_argument(
        "--cache-dir", 
        type=Path,
        help="Directory for caching processed MIDI data"
    )
    
    parser.add_argument(
        "--temp-dir", 
        type=Path,
        help="Directory for temporary files"
    )
    
    # Audio configuration
    audio_group = parser.add_argument_group("Audio Configuration")
    audio_group.add_argument(
        "--sample-rate", 
        type=int, 
        default=16000,
        help="Sample rate for audio processing (default: 16000)"
    )
    
    audio_group.add_argument(
        "--record-duration", 
        type=float, 
        default=60.0,
        help="Duration in seconds for audio recording (default: 60.0)"
    )
    
    audio_group.add_argument(
        "--no-denoise", 
        action="store_true",
        help="Disable audio denoising"
    )
    
    # MIDI configuration
    midi_group = parser.add_argument_group("MIDI Configuration")
    midi_group.add_argument(
        "--samples-per-instrument", 
        type=int, 
        default=10,
        help="Number of samples per instrument for clustering (default: 10)"
    )
    
    midi_group.add_argument(
        "--soundfont-path", 
        type=Path,
        help="Path to soundfont file for MIDI synthesis"
    )
    
    midi_group.add_argument(
        "--max-song-duration", 
        type=float,
        help="Maximum duration of the song in seconds"
    )
    
    # Logging configuration
    logging_group = parser.add_argument_group("Logging Configuration")
    logging_group.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)"
    )
    
    logging_group.add_argument(
        "--log-file", 
        type=Path,
        help="Path to log file (if not provided, logs to console only)"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--save-summary", 
        action="store_true",
        help="Save processing summary to file"
    )
    
    output_group.add_argument(
        "--save-config", 
        action="store_true",
        help="Save the used configuration to file"
    )
    
    output_group.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser


def create_config_from_args(args) -> AmbientMIDIConfig:
    """Create configuration from command line arguments."""
    
    # Load base configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = load_config()
    
    # Override with command line arguments
    if args.sample_rate:
        config.audio.sample_rate = args.sample_rate
    
    if args.record_duration:
        config.audio.record_duration = args.record_duration
    
    if args.no_denoise:
        config.audio.denoise_enabled = False
    
    if args.samples_per_instrument:
        config.midi.samples_per_instrument = args.samples_per_instrument
    
    if args.soundfont_path:
        config.midi.soundfont_path = args.soundfont_path
    
    if args.max_song_duration:
        config.midi.max_song_duration = args.max_song_duration
    
    if args.cache_dir:
        config.paths.cache_dir = args.cache_dir
    
    if args.temp_dir:
        config.paths.temp_dir = args.temp_dir
    
    # Configure logging
    if args.log_level:
        from ambientmidi.config import LogLevel
        config.logging.level = LogLevel(args.log_level)
    
    if args.log_file:
        config.logging.file_handler = args.log_file
    
    if args.verbose:
        config.logging.level = LogLevel.DEBUG
    
    # Re-initialize logging with new configuration
    config.logging.configure_logging()
    
    return config


def progress_callback(step: str, progress: float) -> None:
    """Progress callback for the pipeline."""
    logger = logging.getLogger(__name__)
    
    step_names = {
        "midi_preprocessing": "Processing MIDI file",
        "audio_acquisition": "Acquiring audio",
        "event_processing": "Processing events",
        "audio_rendering": "Rendering audio",
        "saving": "Saving output",
        "complete": "Complete"
    }
    
    step_name = step_names.get(step, step)
    logger.info(f"{step_name}: {progress:.1%}")


def main() -> int:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = create_config_from_args(args)
        logger = logging.getLogger(__name__)
        
        logger.info("Starting AmbientMIDI synthesis")
        logger.info(f"MIDI file: {args.midi_path}")
        logger.info(f"Output file: {args.output_path}")
        
        # Validate inputs
        if not args.midi_path.exists():
            raise FileNotFoundError(f"MIDI file not found: {args.midi_path}")
        
        if not args.midi_path.suffix.lower() in ['.mid', '.midi']:
            raise ValueError(f"Invalid MIDI file extension: {args.midi_path.suffix}")
        
        if args.input_recording and not args.input_recording.exists():
            raise FileNotFoundError(f"Input recording not found: {args.input_recording}")
        
        # Create and run pipeline
        pipeline = create_pipeline(config)
        
        results = pipeline.process(
            midi_path=args.midi_path,
            output_path=args.output_path,
            input_recording_path=args.input_recording,
            progress_callback=progress_callback
        )
        
        # Check if all steps succeeded
        failed_steps = [name for name, result in results.items() if not result.success]
        if failed_steps:
            logger.error(f"Pipeline failed at steps: {', '.join(failed_steps)}")
            return 1
        
        # Save summary if requested
        if args.save_summary:
            summary_path = args.output_path.with_suffix('.summary.json')
            pipeline.save_summary(summary_path)
            logger.info(f"Saved summary to: {summary_path}")
        
        # Save configuration if requested
        if args.save_config:
            config_path = args.output_path.with_suffix('.config.json')
            config.save(config_path)
            logger.info(f"Saved configuration to: {config_path}")
        
        # Print summary
        summary = pipeline.get_summary()
        logger.info(f"Processing completed successfully in {summary['total_processing_time']:.2f}s")
        
        return 0
        
    except AmbientMIDIError as e:
        logger = logging.getLogger(__name__)
        logger.error(f"AmbientMIDI error: {e}")
        if e.details:
            logger.error(f"Error details: {e.details}")
        return 1
        
    except KeyboardInterrupt:
        logger = logging.getLogger(__name__)
        logger.info("Processing interrupted by user")
        return 130  # Standard exit code for SIGINT
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Unexpected error: {e}")
        logger.debug("Exception details:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())