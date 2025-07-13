#!/usr/bin/env python3
"""
Example usage of the refactored AmbientMIDI system.

This script demonstrates various ways to use the new modular, configurable
AmbientMIDI system for processing MIDI files with audio input.
"""

import sys
from pathlib import Path
import logging

# Import the refactored AmbientMIDI system
from ambientmidi import (
    AmbientMIDIConfig,
    create_pipeline,
    process_midi_file,
    load_config,
    create_default_config,
    configure_package_logging,
    get_version_info
)
from ambientmidi.exceptions import AmbientMIDIError


def example_1_basic_usage():
    """Example 1: Basic usage with default settings."""
    print("=== Example 1: Basic Usage ===")
    
    # Check if we have sample files
    midi_path = Path("sample.mid")
    if not midi_path.exists():
        print("No sample MIDI file found. Please provide a MIDI file named 'sample.mid'")
        return
    
    try:
        # Simple processing with defaults
        results = process_midi_file(
            midi_path="sample.mid",
            output_path="output_basic.wav"
        )
        
        # Check results
        if all(result.success for result in results.values()):
            print("✅ Processing completed successfully!")
            print(f"Output saved to: output_basic.wav")
        else:
            print("❌ Processing failed at some steps:")
            for step, result in results.items():
                if not result.success:
                    print(f"  - {step}: {result.error}")
    
    except AmbientMIDIError as e:
        print(f"❌ Error: {e}")
    
    print()


def example_2_with_configuration():
    """Example 2: Using custom configuration."""
    print("=== Example 2: Custom Configuration ===")
    
    # Create custom configuration
    config = create_default_config(
        sample_rate=44100,
        record_duration=30.0,
        samples_per_instrument=15,
        output_dir="custom_output"
    )
    
    # You can also load from file
    # config = load_config("config.json")
    
    # Modify specific settings
    config.audio.denoise_enabled = True
    config.render.harmonic_decay = 0.7
    config.logging.level = config.logging.level.DEBUG
    
    midi_path = Path("sample.mid")
    if not midi_path.exists():
        print("No sample MIDI file found. Please provide a MIDI file named 'sample.mid'")
        return
    
    try:
        results = process_midi_file(
            midi_path="sample.mid",
            output_path="output_custom.wav",
            config=config
        )
        
        if all(result.success for result in results.values()):
            print("✅ Processing completed with custom configuration!")
            print(f"Output saved to: {config.paths.output_dir}/output_custom.wav")
        else:
            print("❌ Processing failed")
    
    except AmbientMIDIError as e:
        print(f"❌ Error: {e}")
    
    print()


def example_3_with_input_audio():
    """Example 3: Using input audio file instead of recording."""
    print("=== Example 3: With Input Audio File ===")
    
    midi_path = Path("sample.mid")
    audio_path = Path("input_audio.wav")
    
    if not midi_path.exists():
        print("No sample MIDI file found. Please provide a MIDI file named 'sample.mid'")
        return
    
    if not audio_path.exists():
        print("No input audio file found. Please provide an audio file named 'input_audio.wav'")
        print("Or remove this example to use audio recording instead.")
        return
    
    try:
        results = process_midi_file(
            midi_path="sample.mid",
            output_path="output_with_audio.wav",
            input_recording_path="input_audio.wav"
        )
        
        if all(result.success for result in results.values()):
            print("✅ Processing completed with input audio!")
            print(f"Output saved to: output_with_audio.wav")
        else:
            print("❌ Processing failed")
    
    except AmbientMIDIError as e:
        print(f"❌ Error: {e}")
    
    print()


def example_4_advanced_pipeline():
    """Example 4: Advanced pipeline usage with progress tracking."""
    print("=== Example 4: Advanced Pipeline Usage ===")
    
    # Create configuration
    config = AmbientMIDIConfig()
    config.audio.sample_rate = 22050
    config.audio.record_duration = 20.0
    config.midi.samples_per_instrument = 8
    
    # Create pipeline
    pipeline = create_pipeline(config)
    
    # Progress callback
    def progress_callback(step, progress):
        step_names = {
            "midi_preprocessing": "🎵 Processing MIDI",
            "audio_acquisition": "🎤 Acquiring audio",
            "event_processing": "🔍 Processing events",
            "audio_rendering": "🎧 Rendering audio",
            "saving": "💾 Saving output",
            "complete": "✅ Complete"
        }
        
        step_name = step_names.get(step, step)
        print(f"[{progress:.1%}] {step_name}")
    
    midi_path = Path("sample.mid")
    if not midi_path.exists():
        print("No sample MIDI file found. Please provide a MIDI file named 'sample.mid'")
        return
    
    try:
        results = pipeline.process(
            midi_path=Path("sample.mid"),
            output_path=Path("output_advanced.wav"),
            progress_callback=progress_callback
        )
        
        # Get detailed summary
        summary = pipeline.get_summary()
        print(f"\n📊 Processing Summary:")
        print(f"Total time: {summary['total_processing_time']:.2f}s")
        print(f"Steps breakdown:")
        
        for step_name, step_info in summary['steps'].items():
            status = "✅" if step_info['success'] else "❌"
            print(f"  {status} {step_name}: {step_info['processing_time']:.2f}s")
            if step_info['error']:
                print(f"    Error: {step_info['error']}")
        
        # Save summary
        pipeline.save_summary(Path("processing_summary.json"))
        print(f"📝 Summary saved to: processing_summary.json")
        
    except AmbientMIDIError as e:
        print(f"❌ Error: {e}")
    
    print()


def example_5_error_handling():
    """Example 5: Comprehensive error handling."""
    print("=== Example 5: Error Handling ===")
    
    from ambientmidi.exceptions import (
        ConfigurationError,
        MIDIProcessingError,
        AudioProcessingError,
        FileNotFoundError
    )
    
    try:
        # This will fail because the file doesn't exist
        results = process_midi_file(
            midi_path="nonexistent.mid",
            output_path="output_error.wav"
        )
        
    except FileNotFoundError as e:
        print(f"📁 File not found: {e}")
        print(f"   Details: {e.details}")
    
    except ConfigurationError as e:
        print(f"⚙️ Configuration error: {e}")
    
    except AudioProcessingError as e:
        print(f"🎤 Audio processing error: {e}")
    
    except MIDIProcessingError as e:
        print(f"🎵 MIDI processing error: {e}")
    
    except AmbientMIDIError as e:
        print(f"⚠️ General AmbientMIDI error: {e}")
    
    print()


def example_6_configuration_file():
    """Example 6: Using configuration file."""
    print("=== Example 6: Configuration File ===")
    
    # Create a sample configuration file
    config_path = Path("example_config.json")
    
    if not config_path.exists():
        print("Creating sample configuration file...")
        config = create_default_config()
        config.save(config_path)
        print(f"📄 Created: {config_path}")
    
    try:
        # Load configuration from file
        config = load_config(config_path)
        print(f"✅ Loaded configuration from: {config_path}")
        
        # Show some configuration values
        print(f"   Sample rate: {config.audio.sample_rate}")
        print(f"   Record duration: {config.audio.record_duration}s")
        print(f"   Samples per instrument: {config.midi.samples_per_instrument}")
        print(f"   Output directory: {config.paths.output_dir}")
        
        # Use the configuration
        midi_path = Path("sample.mid")
        if midi_path.exists():
            results = process_midi_file(
                midi_path="sample.mid",
                output_path="output_from_config.wav",
                config=config
            )
            
            if all(result.success for result in results.values()):
                print("✅ Processing completed with config file!")
        else:
            print("No sample MIDI file found for processing.")
    
    except AmbientMIDIError as e:
        print(f"❌ Error: {e}")
    
    print()


def main():
    """Main function to run all examples."""
    print("🎵 AmbientMIDI v2.0 Examples")
    print("=" * 50)
    
    # Show version info
    version_info = get_version_info()
    print(f"Version: {version_info['version']}")
    print(f"Description: {version_info['description']}")
    print()
    
    # Enable debug logging for demonstrations
    configure_package_logging(logging.INFO)
    
    # Run examples
    try:
        example_1_basic_usage()
        example_2_with_configuration()
        example_3_with_input_audio()
        example_4_advanced_pipeline()
        example_5_error_handling()
        example_6_configuration_file()
        
    except KeyboardInterrupt:
        print("\n⏹️ Examples interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    print("🎉 Examples completed!")
    print("\nTo run individual examples, see the function definitions in this file.")
    print("To get started with your own MIDI files:")
    print("  1. Place a MIDI file named 'sample.mid' in this directory")
    print("  2. Optionally place an audio file named 'input_audio.wav'")
    print("  3. Run this script again")


if __name__ == "__main__":
    main()