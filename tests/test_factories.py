"""
Unit tests for the AmbientMIDI factories module.

Tests cover:
- Factory pattern implementations
- Abstract factory pattern
- Component creation and registration
- Configuration presets and overrides
- Error handling in factories
"""

import unittest
from unittest.mock import patch, MagicMock

from ambientmidi.factories import (
    ProcessingStepFactory,
    ConfigurationFactory,
    AudioProcessorFactory,
    FeatureExtractorFactory,
    ClusteringFactory,
    PipelineFactory,
    AmbientMIDIAbstractFactory,
    StandardAmbientMIDIFactory,
    HighQualityAmbientMIDIFactory,
    FastAmbientMIDIFactory,
    get_factory,
    register_factory,
    create_quick_pipeline,
    create_components
)
from ambientmidi.config import AmbientMIDIConfig
from ambientmidi.pipeline import ProcessingStep
from ambientmidi.exceptions import ConfigurationError
from tests import TestFixtures


class TestProcessingStepFactory(unittest.TestCase):
    """Test ProcessingStepFactory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AmbientMIDIConfig()
        # Store original registry to restore after tests
        self.original_registry = ProcessingStepFactory._step_registry.copy()
    
    def tearDown(self):
        """Restore original registry."""
        ProcessingStepFactory._step_registry = self.original_registry
    
    def test_get_available_steps(self):
        """Test getting available processing steps."""
        steps = ProcessingStepFactory.get_available_steps()
        
        self.assertIn('midi_preprocessing', steps)
        self.assertIn('audio_acquisition', steps)
        self.assertIn('event_processing', steps)
        self.assertIn('audio_rendering', steps)
        self.assertIsInstance(steps, dict)
    
    def test_create_existing_step(self):
        """Test creating an existing processing step."""
        step = ProcessingStepFactory.create_step('midi_preprocessing', self.config)
        
        self.assertEqual(step.name, 'midi_preprocessing')
        self.assertEqual(step.config, self.config)
    
    def test_create_nonexistent_step(self):
        """Test creating a non-existent processing step."""
        with self.assertRaises(ConfigurationError) as cm:
            ProcessingStepFactory.create_step('nonexistent_step', self.config)
        
        self.assertIn("Unknown processing step", str(cm.exception))
        self.assertIn("Available steps", str(cm.exception))
    
    def test_register_valid_step(self):
        """Test registering a valid processing step."""
        class CustomStep(ProcessingStep):
            def process(self, input_data, **kwargs):
                pass
        
        ProcessingStepFactory.register_step('custom_step', CustomStep)
        
        # Should now be able to create the custom step
        step = ProcessingStepFactory.create_step('custom_step', self.config)
        self.assertIsInstance(step, CustomStep)
    
    def test_register_invalid_step(self):
        """Test registering an invalid processing step."""
        class InvalidStep:
            pass
        
        with self.assertRaises(ConfigurationError) as cm:
            ProcessingStepFactory.register_step('invalid_step', InvalidStep)
        
        self.assertIn("Step class must inherit from ProcessingStep", str(cm.exception))
    
    @patch('ambientmidi.config.LoggingConfig.configure_logging')
    def test_create_default_pipeline_steps(self, mock_logging):
        """Test creating all default pipeline steps."""
        steps = ProcessingStepFactory.create_default_pipeline_steps(self.config)
        
        self.assertEqual(len(steps), 4)
        self.assertIn('midi_preprocessing', steps)
        self.assertIn('audio_acquisition', steps)
        self.assertIn('event_processing', steps)
        self.assertIn('audio_rendering', steps)
        
        for step in steps.values():
            self.assertIsInstance(step, ProcessingStep)


class TestConfigurationFactory(unittest.TestCase):
    """Test ConfigurationFactory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Store original presets to restore after tests
        self.original_presets = ConfigurationFactory._preset_configs.copy()
    
    def tearDown(self):
        """Restore original presets."""
        ConfigurationFactory._preset_configs = self.original_presets
    
    @patch('ambientmidi.config.LoggingConfig.configure_logging')
    def test_create_default_config(self, mock_logging):
        """Test creating default configuration."""
        config = ConfigurationFactory.create_config('default')
        
        self.assertIsInstance(config, AmbientMIDIConfig)
        self.assertEqual(config.audio.sample_rate, 16000)  # Default value
    
    @patch('ambientmidi.config.LoggingConfig.configure_logging')
    def test_create_high_quality_config(self, mock_logging):
        """Test creating high-quality configuration."""
        config = ConfigurationFactory.create_config('high_quality')
        
        self.assertIsInstance(config, AmbientMIDIConfig)
        self.assertEqual(config.audio.sample_rate, 44100)
        self.assertEqual(config.spectrogram.n_mels, 80)
        self.assertEqual(config.midi.samples_per_instrument, 20)
    
    @patch('ambientmidi.config.LoggingConfig.configure_logging')
    def test_create_fast_config(self, mock_logging):
        """Test creating fast configuration."""
        config = ConfigurationFactory.create_config('fast')
        
        self.assertIsInstance(config, AmbientMIDIConfig)
        self.assertEqual(config.audio.sample_rate, 16000)
        self.assertEqual(config.spectrogram.n_mels, 20)
        self.assertFalse(config.audio.denoise_enabled)
    
    @patch('ambientmidi.config.LoggingConfig.configure_logging')
    def test_create_config_with_overrides(self, mock_logging):
        """Test creating configuration with overrides."""
        overrides = {
            'audio': {'sample_rate': 48000},
            'midi': {'samples_per_instrument': 15}
        }
        
        config = ConfigurationFactory.create_config('default', **overrides)
        
        self.assertEqual(config.audio.sample_rate, 48000)
        self.assertEqual(config.midi.samples_per_instrument, 15)
    
    def test_create_unknown_preset(self):
        """Test creating configuration with unknown preset."""
        with self.assertRaises(ConfigurationError) as cm:
            ConfigurationFactory.create_config('unknown_preset')
        
        self.assertIn("Unknown preset", str(cm.exception))
        self.assertIn("Available presets", str(cm.exception))
    
    def test_register_preset(self):
        """Test registering a new preset."""
        custom_preset = {
            'audio': {'sample_rate': 96000},
            'midi': {'samples_per_instrument': 50}
        }
        
        ConfigurationFactory.register_preset('custom', custom_preset)
        
        # Should now be able to create config with custom preset
        with patch('ambientmidi.config.LoggingConfig.configure_logging'):
            config = ConfigurationFactory.create_config('custom')
            self.assertEqual(config.audio.sample_rate, 96000)
            self.assertEqual(config.midi.samples_per_instrument, 50)
    
    def test_get_available_presets(self):
        """Test getting available presets."""
        presets = ConfigurationFactory.get_available_presets()
        
        self.assertIn('default', presets)
        self.assertIn('high_quality', presets)
        self.assertIn('fast', presets)
        self.assertIn('production', presets)
        self.assertIsInstance(presets, dict)


class TestAudioProcessorFactory(unittest.TestCase):
    """Test AudioProcessorFactory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AmbientMIDIConfig()
    
    @patch('ambientmidi.events.compute_pcengram')
    def test_create_spectrogram_processor(self, mock_compute_pcengram):
        """Test creating spectrogram processor."""
        mock_audio = TestFixtures.create_mock_audio()
        mock_spectrogram = TestFixtures.create_mock_spectrogram()
        mock_compute_pcengram.return_value = mock_spectrogram
        
        processor = AudioProcessorFactory.create_spectrogram_processor(self.config)
        result = processor(mock_audio)
        
        mock_compute_pcengram.assert_called_once_with(
            mock_audio,
            sample_rate=self.config.audio.sample_rate,
            window_size_ms=self.config.spectrogram.window_size_ms,
            hop_size_ms=self.config.spectrogram.hop_size_ms,
            n_mels=self.config.spectrogram.n_mels,
            power=self.config.spectrogram.power
        )
        self.assertEqual(result, mock_spectrogram)
    
    @patch('ambientmidi.events.get_onsets')
    def test_create_onset_detector(self, mock_get_onsets):
        """Test creating onset detector."""
        mock_spectrogram = TestFixtures.create_mock_spectrogram()
        mock_onsets = ([100, 200], [10, 20], None)
        mock_get_onsets.return_value = mock_onsets
        
        detector = AudioProcessorFactory.create_onset_detector(self.config)
        result = detector(mock_spectrogram)
        
        mock_get_onsets.assert_called_once_with(
            mock_spectrogram,
            sample_rate=self.config.audio.sample_rate,
            hop_size_ms=self.config.spectrogram.hop_size_ms
        )
        self.assertEqual(result, mock_onsets)
    
    @patch('ambientmidi.audio.load_audio')
    def test_create_audio_loader(self, mock_load_audio):
        """Test creating audio loader."""
        mock_audio = TestFixtures.create_mock_audio()
        mock_load_audio.return_value = mock_audio
        
        loader = AudioProcessorFactory.create_audio_loader(self.config)
        result = loader('test_path.wav')
        
        mock_load_audio.assert_called_once_with(
            'test_path.wav',
            sample_rate=self.config.audio.sample_rate
        )
        self.assertEqual(result, mock_audio)
    
    @patch('ambientmidi.audio.record_audio')
    def test_create_audio_recorder(self, mock_record_audio):
        """Test creating audio recorder."""
        mock_audio = TestFixtures.create_mock_audio()
        mock_record_audio.return_value = mock_audio
        
        recorder = AudioProcessorFactory.create_audio_recorder(self.config)
        result = recorder()
        
        mock_record_audio.assert_called_once_with(
            duration=self.config.audio.record_duration,
            sample_rate=self.config.audio.sample_rate,
            denoise=self.config.audio.denoise_enabled
        )
        self.assertEqual(result, mock_audio)


class TestFeatureExtractorFactory(unittest.TestCase):
    """Test FeatureExtractorFactory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AmbientMIDIConfig()
    
    @patch('ambientmidi.features.get_feature_dict')
    def test_create_extractor(self, mock_get_feature_dict):
        """Test creating feature extractor."""
        mock_audio = TestFixtures.create_mock_audio()
        mock_features = {'mfcc': [1, 2, 3], 'pitch': 440}
        mock_get_feature_dict.return_value = mock_features
        
        features = ['mfcc', 'pitch']
        extractor = FeatureExtractorFactory.create_extractor(features, self.config)
        result = extractor(mock_audio)
        
        mock_get_feature_dict.assert_called_once_with(
            mock_audio,
            self.config.audio.sample_rate,
            features=('mfcc', 'pitch')
        )
        self.assertEqual(result, mock_features)
    
    def test_register_extractor(self):
        """Test registering feature extractor."""
        def custom_extractor(audio, sr):
            return {'custom_feature': 42}
        
        FeatureExtractorFactory.register_extractor('custom', custom_extractor)
        
        extractors = FeatureExtractorFactory.get_available_extractors()
        self.assertIn('custom', extractors)
        self.assertEqual(extractors['custom'], custom_extractor)


class TestClusteringFactory(unittest.TestCase):
    """Test ClusteringFactory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AmbientMIDIConfig()
    
    @patch('sklearn_extra.cluster.KMedoids')
    def test_create_kmedoids_clusterer(self, mock_kmedoids):
        """Test creating KMedoids clusterer."""
        self.config.clustering.algorithm = "kmedoids"
        
        clusterer = ClusteringFactory.create_clusterer(self.config)
        result = clusterer(5)
        
        mock_kmedoids.assert_called_once_with(
            n_clusters=5,
            random_state=self.config.clustering.random_state,
            max_iter=self.config.clustering.max_iter
        )
    
    @patch('sklearn.cluster.KMeans')
    def test_create_kmeans_clusterer(self, mock_kmeans):
        """Test creating KMeans clusterer."""
        self.config.clustering.algorithm = "kmeans"
        
        clusterer = ClusteringFactory.create_clusterer(self.config)
        result = clusterer(3)
        
        mock_kmeans.assert_called_once_with(
            n_clusters=3,
            random_state=self.config.clustering.random_state,
            max_iter=self.config.clustering.max_iter,
            tol=self.config.clustering.tol
        )
    
    def test_create_unsupported_clusterer(self):
        """Test creating unsupported clusterer."""
        self.config.clustering.algorithm = "unsupported"
        
        with self.assertRaises(ConfigurationError) as cm:
            ClusteringFactory.create_clusterer(self.config)
        
        self.assertIn("Unsupported clustering algorithm", str(cm.exception))


class TestPipelineFactory(unittest.TestCase):
    """Test PipelineFactory class."""
    
    @patch('ambientmidi.pipeline.AmbientMIDIPipeline')
    @patch('ambientmidi.config.get_default_config')
    def test_create_standard_pipeline_default_config(self, mock_get_default, mock_pipeline):
        """Test creating standard pipeline with default config."""
        mock_config = MagicMock()
        mock_get_default.return_value = mock_config
        
        PipelineFactory.create_standard_pipeline()
        
        mock_get_default.assert_called_once()
        mock_pipeline.assert_called_once_with(mock_config)
    
    @patch('ambientmidi.pipeline.AmbientMIDIPipeline')
    def test_create_standard_pipeline_custom_config(self, mock_pipeline):
        """Test creating standard pipeline with custom config."""
        config = AmbientMIDIConfig()
        
        PipelineFactory.create_standard_pipeline(config)
        
        mock_pipeline.assert_called_once_with(config)
    
    @patch('ambientmidi.pipeline.AmbientMIDIPipeline')
    @patch('ambientmidi.config.LoggingConfig.configure_logging')
    def test_create_preset_pipeline(self, mock_logging, mock_pipeline):
        """Test creating pipeline from preset."""
        PipelineFactory.create_preset_pipeline('fast', audio={'sample_rate': 22050})
        
        mock_pipeline.assert_called_once()
        # Verify config was created with preset and overrides
        call_args = mock_pipeline.call_args[0]
        config = call_args[0]
        self.assertEqual(config.audio.sample_rate, 22050)


class TestAbstractFactories(unittest.TestCase):
    """Test abstract factory implementations."""
    
    @patch('ambientmidi.config.LoggingConfig.configure_logging')
    def test_standard_factory(self, mock_logging):
        """Test StandardAmbientMIDIFactory."""
        factory = StandardAmbientMIDIFactory()
        
        # Test config creation
        config = factory.create_config()
        self.assertIsInstance(config, AmbientMIDIConfig)
        
        # Test pipeline creation
        with patch('ambientmidi.pipeline.AmbientMIDIPipeline') as mock_pipeline:
            factory.create_pipeline(config)
            mock_pipeline.assert_called_once_with(config)
        
        # Test audio processor creation
        processors = factory.create_audio_processor(config)
        self.assertIn('spectrogram', processors)
        self.assertIn('onset_detector', processors)
        self.assertIn('loader', processors)
        self.assertIn('recorder', processors)
    
    @patch('ambientmidi.config.LoggingConfig.configure_logging')
    def test_high_quality_factory(self, mock_logging):
        """Test HighQualityAmbientMIDIFactory."""
        factory = HighQualityAmbientMIDIFactory()
        
        config = factory.create_config()
        self.assertEqual(config.audio.sample_rate, 44100)
        self.assertEqual(config.spectrogram.n_mels, 80)
    
    @patch('ambientmidi.config.LoggingConfig.configure_logging')
    def test_fast_factory(self, mock_logging):
        """Test FastAmbientMIDIFactory."""
        factory = FastAmbientMIDIFactory()
        
        config = factory.create_config()
        self.assertEqual(config.audio.sample_rate, 16000)
        self.assertFalse(config.audio.denoise_enabled)


class TestFactoryRegistry(unittest.TestCase):
    """Test factory registry functions."""
    
    def test_get_standard_factory(self):
        """Test getting standard factory."""
        factory = get_factory('standard')
        self.assertIsInstance(factory, StandardAmbientMIDIFactory)
    
    def test_get_high_quality_factory(self):
        """Test getting high-quality factory."""
        factory = get_factory('high_quality')
        self.assertIsInstance(factory, HighQualityAmbientMIDIFactory)
    
    def test_get_fast_factory(self):
        """Test getting fast factory."""
        factory = get_factory('fast')
        self.assertIsInstance(factory, FastAmbientMIDIFactory)
    
    def test_get_unknown_factory(self):
        """Test getting unknown factory type."""
        with self.assertRaises(ConfigurationError) as cm:
            get_factory('unknown')
        
        self.assertIn("Unknown factory type", str(cm.exception))
        self.assertIn("Available types", str(cm.exception))
    
    def test_register_factory(self):
        """Test registering new factory."""
        class CustomFactory(AmbientMIDIAbstractFactory):
            def create_config(self):
                return AmbientMIDIConfig()
            
            def create_pipeline(self, config):
                pass
            
            def create_audio_processor(self, config):
                pass
        
        register_factory('custom', CustomFactory)
        
        factory = get_factory('custom')
        self.assertIsInstance(factory, CustomFactory)
    
    def test_register_invalid_factory(self):
        """Test registering invalid factory."""
        class InvalidFactory:
            pass
        
        with self.assertRaises(ConfigurationError) as cm:
            register_factory('invalid', InvalidFactory)
        
        self.assertIn("Factory must inherit from AmbientMIDIAbstractFactory", str(cm.exception))


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    @patch('ambientmidi.factories.PipelineFactory.create_preset_pipeline')
    def test_create_quick_pipeline(self, mock_create_preset):
        """Test create_quick_pipeline convenience function."""
        create_quick_pipeline('fast', audio={'sample_rate': 22050})
        
        mock_create_preset.assert_called_once_with('fast', audio={'sample_rate': 22050})
    
    @patch('ambientmidi.config.LoggingConfig.configure_logging')
    def test_create_components(self, mock_logging):
        """Test create_components convenience function."""
        with patch('ambientmidi.pipeline.AmbientMIDIPipeline') as mock_pipeline:
            components = create_components('standard')
            
            self.assertIn('config', components)
            self.assertIn('pipeline', components)
            self.assertIn('audio_processor', components)
            self.assertIn('factory', components)
            
            self.assertIsInstance(components['config'], AmbientMIDIConfig)
            self.assertIsInstance(components['factory'], StandardAmbientMIDIFactory)


if __name__ == '__main__':
    unittest.main()