"""
Unit tests for the AmbientMIDI pipeline module.

Tests cover:
- Individual processing steps
- Pipeline orchestration and flow
- Error handling and recovery
- Progress tracking and callbacks
- Integration between components
"""

import unittest
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from ambientmidi.pipeline import (
    ProcessingStep,
    ProcessingResult,
    MIDIPreprocessingStep,
    AudioAcquisitionStep,
    EventProcessingStep,
    AudioRenderingStep,
    AmbientMIDIPipeline,
    create_pipeline
)
from ambientmidi.config import AmbientMIDIConfig
from ambientmidi.exceptions import (
    MIDIProcessingError,
    AudioProcessingError,
    RenderingError,
    FileNotFoundError
)
from tests import TestFixtures


class TestProcessingResult(unittest.TestCase):
    """Test ProcessingResult class."""
    
    def test_successful_result_creation(self):
        """Test creating a successful processing result."""
        data = {"test": "data"}
        metadata = {"source": "test"}
        processing_time = 1.5
        
        result = ProcessingResult(
            data=data,
            metadata=metadata,
            processing_time=processing_time,
            success=True
        )
        
        self.assertEqual(result.data, data)
        self.assertEqual(result.metadata, metadata)
        self.assertEqual(result.processing_time, processing_time)
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
    
    def test_failed_result_creation(self):
        """Test creating a failed processing result."""
        error = MIDIProcessingError("Test error")
        processing_time = 0.5
        
        result = ProcessingResult(
            data=None,
            metadata={"source": "test"},
            processing_time=processing_time,
            success=False,
            error=error
        )
        
        self.assertIsNone(result.data)
        self.assertEqual(result.processing_time, processing_time)
        self.assertFalse(result.success)
        self.assertEqual(result.error, error)


class TestProcessingStep(unittest.TestCase):
    """Test abstract ProcessingStep class."""
    
    def test_abstract_instantiation(self):
        """Test that ProcessingStep cannot be instantiated directly."""
        config = AmbientMIDIConfig()
        
        with self.assertRaises(TypeError):
            ProcessingStep("test", config)
    
    def test_subclass_implementation(self):
        """Test that subclass must implement process method."""
        config = AmbientMIDIConfig()
        
        class TestStep(ProcessingStep):
            pass
        
        with self.assertRaises(TypeError):
            TestStep("test", config)
        
        class ValidTestStep(ProcessingStep):
            def process(self, input_data, **kwargs):
                return ProcessingResult(
                    data=input_data,
                    metadata={},
                    processing_time=0.1,
                    success=True
                )
        
        # Should not raise exception
        step = ValidTestStep("test", config)
        self.assertEqual(step.name, "test")
        self.assertEqual(step.config, config)


class TestMIDIPreprocessingStep(unittest.TestCase):
    """Test MIDIPreprocessingStep class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AmbientMIDIConfig()
        self.step = MIDIPreprocessingStep(self.config)
        self.temp_dir = TestFixtures.create_temp_dir()
        
    def tearDown(self):
        """Clean up test fixtures."""
        TestFixtures.cleanup_temp_dir(self.temp_dir)
    
    def test_step_initialization(self):
        """Test step initialization."""
        self.assertEqual(self.step.name, "midi_preprocessing")
        self.assertEqual(self.step.config, self.config)
    
    def test_validate_input_valid_file(self):
        """Test input validation with valid MIDI file."""
        midi_file = self.temp_dir / "test.mid"
        midi_file.touch()
        
        # Should not raise exception
        self.step.validate_input(midi_file)
    
    def test_validate_input_nonexistent_file(self):
        """Test input validation with nonexistent file."""
        midi_file = self.temp_dir / "nonexistent.mid"
        
        with self.assertRaises(FileNotFoundError):
            self.step.validate_input(midi_file)
    
    def test_validate_input_invalid_extension(self):
        """Test input validation with invalid file extension."""
        audio_file = self.temp_dir / "test.wav"
        audio_file.touch()
        
        with self.assertRaises(MIDIProcessingError) as cm:
            self.step.validate_input(audio_file)
        self.assertIn("Invalid MIDI file extension", str(cm.exception))
    
    @patch('ambientmidi.pipeline.json.load')
    @patch('pathlib.Path.exists')
    def test_process_with_cache(self, mock_exists, mock_json_load):
        """Test processing with existing cache."""
        midi_file = self.temp_dir / "test.mid"
        midi_file.touch()
        
        # Mock cache exists
        mock_exists.return_value = True
        mock_midi_info = TestFixtures.create_mock_midi_info()
        mock_json_load.return_value = mock_midi_info
        
        result = self.step.process(midi_file)
        
        self.assertTrue(result.success)
        self.assertEqual(result.data, mock_midi_info)
        self.assertTrue(result.metadata["cached"])
        mock_json_load.assert_called_once()
    
    @patch('ambientmidi.pipeline.json.dump')
    @patch('ambientmidi.midi.preprocess_midi_file')
    @patch('pathlib.Path.exists')
    def test_process_without_cache(self, mock_exists, mock_preprocess, mock_json_dump):
        """Test processing without existing cache."""
        midi_file = self.temp_dir / "test.mid"
        midi_file.touch()
        
        # Mock cache doesn't exist
        mock_exists.return_value = False
        mock_midi_info = TestFixtures.create_mock_midi_info()
        mock_preprocess.return_value = mock_midi_info
        
        result = self.step.process(midi_file)
        
        self.assertTrue(result.success)
        self.assertEqual(result.data, mock_midi_info)
        self.assertFalse(result.metadata["cached"])
        mock_preprocess.assert_called_once()
        mock_json_dump.assert_called_once()
    
    @patch('ambientmidi.midi.preprocess_midi_file')
    def test_process_error_handling(self, mock_preprocess):
        """Test error handling during processing."""
        midi_file = self.temp_dir / "test.mid"
        midi_file.touch()
        
        # Mock preprocessing failure
        mock_preprocess.side_effect = Exception("Preprocessing failed")
        
        result = self.step.process(midi_file)
        
        self.assertFalse(result.success)
        self.assertIsNone(result.data)
        self.assertIsNotNone(result.error)


class TestAudioAcquisitionStep(unittest.TestCase):
    """Test AudioAcquisitionStep class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AmbientMIDIConfig()
        self.step = AudioAcquisitionStep(self.config)
        self.temp_dir = TestFixtures.create_temp_dir()
        
    def tearDown(self):
        """Clean up test fixtures."""
        TestFixtures.cleanup_temp_dir(self.temp_dir)
    
    def test_step_initialization(self):
        """Test step initialization."""
        self.assertEqual(self.step.name, "audio_acquisition")
        self.assertEqual(self.step.config, self.config)
    
    def test_validate_input_valid_file(self):
        """Test input validation with valid audio file."""
        audio_file = self.temp_dir / "test.wav"
        audio_file.touch()
        
        # Should not raise exception
        self.step.validate_input(audio_file)
    
    def test_validate_input_nonexistent_file(self):
        """Test input validation with nonexistent file."""
        audio_file = self.temp_dir / "nonexistent.wav"
        
        with self.assertRaises(FileNotFoundError):
            self.step.validate_input(audio_file)
    
    @patch('ambientmidi.audio.load_audio')
    def test_process_with_file(self, mock_load_audio):
        """Test processing with input file."""
        audio_file = self.temp_dir / "test.wav"
        audio_file.touch()
        
        mock_audio = TestFixtures.create_mock_audio()
        mock_load_audio.return_value = mock_audio
        
        result = self.step.process(input_recording_path=audio_file)
        
        self.assertTrue(result.success)
        self.assertEqual(result.data.shape, mock_audio.shape)
        self.assertEqual(result.metadata["source"], "file")
        self.assertEqual(result.metadata["file_path"], str(audio_file))
        mock_load_audio.assert_called_once_with(audio_file, self.config.audio.sample_rate)
    
    @patch('ambientmidi.audio.record_audio')
    def test_process_with_recording(self, mock_record_audio):
        """Test processing with recording."""
        mock_audio = TestFixtures.create_mock_audio()
        mock_record_audio.return_value = mock_audio
        
        result = self.step.process()
        
        self.assertTrue(result.success)
        self.assertEqual(result.data.shape, mock_audio.shape)
        self.assertEqual(result.metadata["source"], "recording")
        self.assertEqual(result.metadata["duration"], self.config.audio.record_duration)
        mock_record_audio.assert_called_once()
    
    @patch('ambientmidi.audio.load_audio')
    def test_process_error_handling(self, mock_load_audio):
        """Test error handling during processing."""
        audio_file = self.temp_dir / "test.wav"
        audio_file.touch()
        
        # Mock loading failure
        mock_load_audio.side_effect = Exception("Loading failed")
        
        result = self.step.process(input_recording_path=audio_file)
        
        self.assertFalse(result.success)
        self.assertIsNone(result.data)
        self.assertIsNotNone(result.error)


class TestEventProcessingStep(unittest.TestCase):
    """Test EventProcessingStep class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AmbientMIDIConfig()
        self.step = EventProcessingStep(self.config)
        self.mock_audio = TestFixtures.create_mock_audio()
        
    def test_step_initialization(self):
        """Test step initialization."""
        self.assertEqual(self.step.name, "event_processing")
        self.assertEqual(self.step.config, self.config)
    
    @patch('ambientmidi.cluster.get_clip_clusters')
    @patch('ambientmidi.events.get_event_clip_dicts')
    @patch('ambientmidi.events.get_onsets')
    @patch('ambientmidi.events.compute_pcengram')
    def test_process_success(self, mock_pcengram, mock_onsets, mock_clips, mock_clusters):
        """Test successful event processing."""
        # Mock the processing chain
        mock_pcengram.return_value = TestFixtures.create_mock_spectrogram()
        mock_onsets.return_value = ([100, 200, 300], [10, 20, 30], None)
        mock_clips.return_value = [{"test": "clip1"}, {"test": "clip2"}]
        mock_clusters.return_value = {"cluster1": {"events": []}}
        
        result = self.step.process(self.mock_audio)
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.data)
        self.assertEqual(result.metadata["num_onsets"], 3)
        self.assertEqual(result.metadata["num_events"], 2)
        self.assertEqual(result.metadata["num_clusters"], 1)
        
        # Verify function calls
        mock_pcengram.assert_called_once_with(self.mock_audio, sample_rate=self.config.audio.sample_rate)
        mock_onsets.assert_called_once()
        mock_clips.assert_called_once()
        mock_clusters.assert_called_once()
    
    @patch('ambientmidi.events.compute_pcengram')
    def test_process_error_handling(self, mock_pcengram):
        """Test error handling during processing."""
        # Mock processing failure
        mock_pcengram.side_effect = Exception("PCEN computation failed")
        
        result = self.step.process(self.mock_audio)
        
        self.assertFalse(result.success)
        self.assertIsNone(result.data)
        self.assertIsNotNone(result.error)


class TestAudioRenderingStep(unittest.TestCase):
    """Test AudioRenderingStep class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AmbientMIDIConfig()
        self.step = AudioRenderingStep(self.config)
        self.mock_midi_info = TestFixtures.create_mock_midi_info()
        self.mock_clusters = {"cluster1": {"events": []}}
        
    def test_step_initialization(self):
        """Test step initialization."""
        self.assertEqual(self.step.name, "audio_rendering")
        self.assertEqual(self.step.config, self.config)
    
    @patch('ambientmidi.render.render_song_from_events')
    def test_process_success(self, mock_render):
        """Test successful audio rendering."""
        mock_audio_output = TestFixtures.create_mock_audio(duration=30.0)
        mock_render.return_value = mock_audio_output
        
        result = self.step.process(self.mock_midi_info, self.mock_clusters)
        
        self.assertTrue(result.success)
        self.assertEqual(result.data.shape, mock_audio_output.shape)
        self.assertAlmostEqual(result.metadata["duration"], 30.0, places=1)
        self.assertEqual(result.metadata["sample_rate"], self.config.audio.sample_rate)
        
        mock_render.assert_called_once_with(
            self.mock_midi_info,
            self.mock_clusters,
            max_song_duration=self.config.midi.max_song_duration,
            sample_rate=self.config.audio.sample_rate
        )
    
    @patch('ambientmidi.render.render_song_from_events')
    def test_process_error_handling(self, mock_render):
        """Test error handling during rendering."""
        # Mock rendering failure
        mock_render.side_effect = Exception("Rendering failed")
        
        result = self.step.process(self.mock_midi_info, self.mock_clusters)
        
        self.assertFalse(result.success)
        self.assertIsNone(result.data)
        self.assertIsNotNone(result.error)


class TestAmbientMIDIPipeline(unittest.TestCase):
    """Test AmbientMIDIPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AmbientMIDIConfig()
        self.pipeline = AmbientMIDIPipeline(self.config)
        self.temp_dir = TestFixtures.create_temp_dir()
        
    def tearDown(self):
        """Clean up test fixtures."""
        TestFixtures.cleanup_temp_dir(self.temp_dir)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertEqual(self.pipeline.config, self.config)
        self.assertIn("midi_preprocessing", self.pipeline.steps)
        self.assertIn("audio_acquisition", self.pipeline.steps)
        self.assertIn("event_processing", self.pipeline.steps)
        self.assertIn("audio_rendering", self.pipeline.steps)
        self.assertEqual(len(self.pipeline.results), 0)
    
    @patch('soundfile.write')
    def test_successful_pipeline_execution(self, mock_sf_write):
        """Test successful end-to-end pipeline execution."""
        # Create test files
        midi_file = self.temp_dir / "test.mid"
        output_file = self.temp_dir / "output.wav"
        midi_file.touch()
        
        # Mock all processing steps
        mock_midi_info = TestFixtures.create_mock_midi_info()
        mock_audio = TestFixtures.create_mock_audio()
        mock_clusters = {"cluster1": {"events": []}}
        mock_output_audio = TestFixtures.create_mock_audio(duration=30.0)
        
        with patch.object(self.pipeline.steps["midi_preprocessing"], "process") as mock_midi:
            with patch.object(self.pipeline.steps["audio_acquisition"], "process") as mock_audio_acq:
                with patch.object(self.pipeline.steps["event_processing"], "process") as mock_events:
                    with patch.object(self.pipeline.steps["audio_rendering"], "process") as mock_render:
                        
                        # Configure mock returns
                        mock_midi.return_value = ProcessingResult(
                            data=mock_midi_info, metadata={}, processing_time=1.0, success=True
                        )
                        mock_audio_acq.return_value = ProcessingResult(
                            data=mock_audio, metadata={}, processing_time=0.5, success=True
                        )
                        mock_events.return_value = ProcessingResult(
                            data=mock_clusters, metadata={}, processing_time=2.0, success=True
                        )
                        mock_render.return_value = ProcessingResult(
                            data=mock_output_audio, metadata={}, processing_time=1.5, success=True
                        )
                        
                        # Run pipeline
                        results = self.pipeline.process(midi_file, output_file)
                        
                        # Verify all steps were executed
                        self.assertEqual(len(results), 4)
                        self.assertTrue(all(result.success for result in results.values()))
                        
                        # Verify output was saved
                        mock_sf_write.assert_called_once()
    
    def test_pipeline_failure_early_step(self):
        """Test pipeline failure in early step."""
        # Create test files
        midi_file = self.temp_dir / "test.mid"
        output_file = self.temp_dir / "output.wav"
        midi_file.touch()
        
        # Mock MIDI preprocessing failure
        with patch.object(self.pipeline.steps["midi_preprocessing"], "process") as mock_midi:
            mock_midi.return_value = ProcessingResult(
                data=None, metadata={}, processing_time=0.1, 
                success=False, error=MIDIProcessingError("MIDI failed")
            )
            
            results = self.pipeline.process(midi_file, output_file)
            
            # Should only have MIDI preprocessing result
            self.assertEqual(len(results), 1)
            self.assertFalse(results["midi_preprocessing"].success)
    
    def test_progress_callback(self):
        """Test progress callback functionality."""
        # Create test files
        midi_file = self.temp_dir / "test.mid"
        output_file = self.temp_dir / "output.wav"
        midi_file.touch()
        
        progress_updates = []
        
        def progress_callback(step, progress):
            progress_updates.append((step, progress))
        
        # Mock all steps to succeed
        mock_results = [
            ProcessingResult(data={}, metadata={}, processing_time=0.1, success=True),
            ProcessingResult(data=TestFixtures.create_mock_audio(), metadata={}, processing_time=0.1, success=True),
            ProcessingResult(data={}, metadata={}, processing_time=0.1, success=True),
            ProcessingResult(data=TestFixtures.create_mock_audio(), metadata={}, processing_time=0.1, success=True)
        ]
        
        with patch.object(self.pipeline.steps["midi_preprocessing"], "process", return_value=mock_results[0]):
            with patch.object(self.pipeline.steps["audio_acquisition"], "process", return_value=mock_results[1]):
                with patch.object(self.pipeline.steps["event_processing"], "process", return_value=mock_results[2]):
                    with patch.object(self.pipeline.steps["audio_rendering"], "process", return_value=mock_results[3]):
                        with patch('soundfile.write'):
                            
                            self.pipeline.process(midi_file, output_file, progress_callback=progress_callback)
                            
                            # Verify progress updates
                            self.assertGreater(len(progress_updates), 0)
                            self.assertEqual(progress_updates[-1], ("complete", 1.0))
    
    def test_get_summary(self):
        """Test pipeline summary generation."""
        # Add some mock results
        self.pipeline.results["test_step"] = ProcessingResult(
            data={"test": "data"},
            metadata={"source": "test"},
            processing_time=1.5,
            success=True
        )
        
        summary = self.pipeline.get_summary()
        
        self.assertIn("total_processing_time", summary)
        self.assertIn("steps", summary)
        self.assertEqual(summary["total_processing_time"], 1.5)
        self.assertIn("test_step", summary["steps"])
        self.assertTrue(summary["steps"]["test_step"]["success"])
    
    def test_save_summary(self):
        """Test saving pipeline summary to file."""
        summary_file = self.temp_dir / "summary.json"
        
        # Add a mock result
        self.pipeline.results["test"] = ProcessingResult(
            data={}, metadata={}, processing_time=1.0, success=True
        )
        
        self.pipeline.save_summary(summary_file)
        
        self.assertTrue(summary_file.exists())
        
        # Verify file contents
        with open(summary_file, 'r') as f:
            saved_summary = json.load(f)
        
        self.assertIn("total_processing_time", saved_summary)
        self.assertIn("steps", saved_summary)


class TestPipelineUtilities(unittest.TestCase):
    """Test pipeline utility functions."""
    
    @patch('ambientmidi.config.get_default_config')
    def test_create_pipeline_default_config(self, mock_get_default):
        """Test creating pipeline with default config."""
        mock_config = AmbientMIDIConfig()
        mock_get_default.return_value = mock_config
        
        pipeline = create_pipeline()
        
        self.assertIsInstance(pipeline, AmbientMIDIPipeline)
        mock_get_default.assert_called_once()
    
    def test_create_pipeline_custom_config(self):
        """Test creating pipeline with custom config."""
        config = AmbientMIDIConfig()
        config.audio.sample_rate = 44100
        
        pipeline = create_pipeline(config)
        
        self.assertIsInstance(pipeline, AmbientMIDIPipeline)
        self.assertEqual(pipeline.config.audio.sample_rate, 44100)


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for pipeline components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = TestFixtures.create_temp_dir()
        self.config = AmbientMIDIConfig()
        
    def tearDown(self):
        """Clean up test fixtures."""
        TestFixtures.cleanup_temp_dir(self.temp_dir)
    
    @patch('ambientmidi.render.render_song_from_events')
    @patch('ambientmidi.cluster.get_clip_clusters')
    @patch('ambientmidi.events.get_event_clip_dicts')
    @patch('ambientmidi.events.get_onsets')
    @patch('ambientmidi.events.compute_pcengram')
    @patch('ambientmidi.audio.record_audio')
    @patch('ambientmidi.midi.preprocess_midi_file')
    @patch('soundfile.write')
    def test_end_to_end_pipeline(self, mock_sf_write, mock_preprocess, mock_record,
                                 mock_pcengram, mock_onsets, mock_clips, mock_clusters, mock_render):
        """Test end-to-end pipeline execution with mocked dependencies."""
        
        # Create test files
        midi_file = self.temp_dir / "test.mid"
        output_file = self.temp_dir / "output.wav"
        midi_file.touch()
        
        # Mock all processing functions
        mock_preprocess.return_value = TestFixtures.create_mock_midi_info()
        mock_record.return_value = TestFixtures.create_mock_audio()
        mock_pcengram.return_value = TestFixtures.create_mock_spectrogram()
        mock_onsets.return_value = ([100, 200], [10, 20], None)
        mock_clips.return_value = [{"test": "clip"}]
        mock_clusters.return_value = {"cluster": {"events": []}}
        mock_render.return_value = TestFixtures.create_mock_audio(duration=30.0)
        
        # Create and run pipeline
        pipeline = create_pipeline(self.config)
        results = pipeline.process(midi_file, output_file)
        
        # Verify all steps completed successfully
        self.assertEqual(len(results), 4)
        self.assertTrue(all(result.success for result in results.values()))
        
        # Verify all functions were called
        mock_preprocess.assert_called_once()
        mock_record.assert_called_once()
        mock_pcengram.assert_called_once()
        mock_onsets.assert_called_once()
        mock_clips.assert_called_once()
        mock_clusters.assert_called_once()
        mock_render.assert_called_once()
        mock_sf_write.assert_called_once()
    
    def test_pipeline_error_propagation(self):
        """Test that errors propagate correctly through pipeline."""
        midi_file = self.temp_dir / "nonexistent.mid"  # File doesn't exist
        output_file = self.temp_dir / "output.wav"
        
        pipeline = create_pipeline(self.config)
        results = pipeline.process(midi_file, output_file)
        
        # MIDI preprocessing should fail
        self.assertFalse(results["midi_preprocessing"].success)
        self.assertIsInstance(results["midi_preprocessing"].error, FileNotFoundError)
        
        # Only MIDI preprocessing should have run
        self.assertEqual(len(results), 1)


if __name__ == '__main__':
    unittest.main()