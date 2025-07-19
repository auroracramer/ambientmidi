"""
Factory patterns for AmbientMIDI components.

This module provides factory classes for creating various components
of the AmbientMIDI system in a flexible and maintainable way.
"""

from abc import ABC, abstractmethod
from typing import Dict, Type, Any, Optional
import logging

from .config import AmbientMIDIConfig
from .pipeline import (
    ProcessingStep, 
    MIDIPreprocessingStep, 
    AudioAcquisitionStep,
    EventProcessingStep, 
    AudioRenderingStep
)
from .exceptions import ConfigurationError, AmbientMIDIError


logger = logging.getLogger(__name__)


class ProcessingStepFactory:
    """Factory for creating processing steps."""
    
    _step_registry: Dict[str, Type[ProcessingStep]] = {
        'midi_preprocessing': MIDIPreprocessingStep,
        'audio_acquisition': AudioAcquisitionStep,
        'event_processing': EventProcessingStep,
        'audio_rendering': AudioRenderingStep,
    }
    
    @classmethod
    def register_step(cls, name: str, step_class: Type[ProcessingStep]) -> None:
        """Register a new processing step type."""
        if not issubclass(step_class, ProcessingStep):
            raise ConfigurationError(f"Step class must inherit from ProcessingStep")
        
        cls._step_registry[name] = step_class
        logger.debug(f"Registered processing step: {name}")
    
    @classmethod
    def create_step(cls, name: str, config: AmbientMIDIConfig, **kwargs) -> ProcessingStep:
        """Create a processing step by name."""
        if name not in cls._step_registry:
            available_steps = list(cls._step_registry.keys())
            raise ConfigurationError(
                f"Unknown processing step: {name}. "
                f"Available steps: {available_steps}"
            )
        
        step_class = cls._step_registry[name]
        
        try:
            return step_class(config, **kwargs)
        except Exception as e:
            raise ConfigurationError(f"Failed to create step '{name}': {e}")
    
    @classmethod
    def get_available_steps(cls) -> Dict[str, Type[ProcessingStep]]:
        """Get all available processing steps."""
        return cls._step_registry.copy()
    
    @classmethod
    def create_default_pipeline_steps(cls, config: AmbientMIDIConfig) -> Dict[str, ProcessingStep]:
        """Create all default pipeline steps."""
        steps = {}
        
        for step_name in ['midi_preprocessing', 'audio_acquisition', 'event_processing', 'audio_rendering']:
            try:
                steps[step_name] = cls.create_step(step_name, config)
                logger.debug(f"Created step: {step_name}")
            except Exception as e:
                logger.error(f"Failed to create step '{step_name}': {e}")
                raise
        
        return steps


class ConfigurationFactory:
    """Factory for creating and managing configurations."""
    
    _preset_configs: Dict[str, Dict[str, Any]] = {
        'default': {},
        'high_quality': {
            'audio': {
                'sample_rate': 44100,
                'target_db_lufs': -16.0,
                'record_duration': 120.0
            },
            'spectrogram': {
                'n_mels': 80,
                'window_size_ms': 12.5
            },
            'midi': {
                'samples_per_instrument': 20
            }
        },
        'fast': {
            'audio': {
                'sample_rate': 16000,
                'record_duration': 30.0,
                'denoise_enabled': False
            },
            'spectrogram': {
                'n_mels': 20,
                'window_size_ms': 50.0
            },
            'midi': {
                'samples_per_instrument': 5
            }
        },
        'production': {
            'audio': {
                'sample_rate': 48000,
                'target_db_lufs': -14.0,
                'use_peak_normalization': False,
                'record_duration': 180.0
            },
            'spectrogram': {
                'n_mels': 128,
                'window_size_ms': 10.0,
                'hop_size_ms': 5.0
            },
            'midi': {
                'samples_per_instrument': 30
            },
            'render': {
                'num_harmonics': 8,
                'resonance_quality': 60.0
            }
        }
    }
    
    @classmethod
    def register_preset(cls, name: str, config_dict: Dict[str, Any]) -> None:
        """Register a new configuration preset."""
        cls._preset_configs[name] = config_dict
        logger.debug(f"Registered configuration preset: {name}")
    
    @classmethod
    def create_config(cls, preset: str = 'default', **overrides) -> AmbientMIDIConfig:
        """Create a configuration from a preset with optional overrides."""
        if preset not in cls._preset_configs:
            available_presets = list(cls._preset_configs.keys())
            raise ConfigurationError(
                f"Unknown preset: {preset}. "
                f"Available presets: {available_presets}"
            )
        
        # Start with the preset
        config_dict = cls._preset_configs[preset].copy()
        
        # Apply overrides
        if overrides:
            cls._deep_update(config_dict, overrides)
        
        # Create configuration
        if config_dict:
            return AmbientMIDIConfig.from_dict(config_dict)
        else:
            return AmbientMIDIConfig()
    
    @classmethod
    def get_available_presets(cls) -> Dict[str, Dict[str, Any]]:
        """Get all available configuration presets."""
        return cls._preset_configs.copy()
    
    @classmethod
    def _deep_update(cls, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """Deep update a dictionary with another dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                cls._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value


class AudioProcessorFactory:
    """Factory for creating audio processing components."""
    
    @staticmethod
    def create_spectrogram_processor(config: AmbientMIDIConfig):
        """Create a spectrogram processor with the given configuration."""
        from .events import compute_pcengram
        
        def processor(audio):
            return compute_pcengram(
                audio,
                sample_rate=config.audio.sample_rate,
                window_size_ms=config.spectrogram.window_size_ms,
                hop_size_ms=config.spectrogram.hop_size_ms,
                n_mels=config.spectrogram.n_mels,
                power=config.spectrogram.power
            )
        
        return processor
    
    @staticmethod
    def create_onset_detector(config: AmbientMIDIConfig):
        """Create an onset detector with the given configuration."""
        from .events import get_onsets
        
        def detector(pcengram):
            return get_onsets(
                pcengram,
                sample_rate=config.audio.sample_rate,
                hop_size_ms=config.spectrogram.hop_size_ms
            )
        
        return detector
    
    @staticmethod
    def create_audio_loader(config: AmbientMIDIConfig):
        """Create an audio loader with the given configuration."""
        from .audio import load_audio
        
        def loader(path, **kwargs):
            return load_audio(
                path,
                sample_rate=config.audio.sample_rate,
                **kwargs
            )
        
        return loader
    
    @staticmethod
    def create_audio_recorder(config: AmbientMIDIConfig):
        """Create an audio recorder with the given configuration."""
        from .audio import record_audio
        
        def recorder(**kwargs):
            return record_audio(
                duration=config.audio.record_duration,
                sample_rate=config.audio.sample_rate,
                denoise=config.audio.denoise_enabled,
                **kwargs
            )
        
        return recorder


class FeatureExtractorFactory:
    """Factory for creating feature extractors."""
    
    _extractor_registry: Dict[str, callable] = {}
    
    @classmethod
    def register_extractor(cls, name: str, extractor_func: callable) -> None:
        """Register a feature extractor function."""
        cls._extractor_registry[name] = extractor_func
        logger.debug(f"Registered feature extractor: {name}")
    
    @classmethod
    def create_extractor(cls, features: list, config: AmbientMIDIConfig):
        """Create a feature extractor for the specified features."""
        from .features import get_feature_dict
        
        def extractor(audio):
            return get_feature_dict(
                audio, 
                config.audio.sample_rate, 
                features=tuple(features)
            )
        
        return extractor
    
    @classmethod
    def get_available_extractors(cls) -> Dict[str, callable]:
        """Get all available feature extractors."""
        return cls._extractor_registry.copy()


class ClusteringFactory:
    """Factory for creating clustering algorithms."""
    
    @staticmethod
    def create_clusterer(config: AmbientMIDIConfig):
        """Create a clusterer with the given configuration."""
        if config.clustering.algorithm == "kmedoids":
            from sklearn_extra.cluster import KMedoids
            
            def clusterer(n_clusters):
                return KMedoids(
                    n_clusters=n_clusters,
                    random_state=config.clustering.random_state,
                    max_iter=config.clustering.max_iter
                )
        
        elif config.clustering.algorithm == "kmeans":
            from sklearn.cluster import KMeans
            
            def clusterer(n_clusters):
                return KMeans(
                    n_clusters=n_clusters,
                    random_state=config.clustering.random_state,
                    max_iter=config.clustering.max_iter,
                    tol=config.clustering.tol
                )
        
        else:
            raise ConfigurationError(f"Unsupported clustering algorithm: {config.clustering.algorithm}")
        
        return clusterer


class PipelineFactory:
    """Factory for creating complete processing pipelines."""
    
    @staticmethod
    def create_standard_pipeline(config: Optional[AmbientMIDIConfig] = None) -> 'AmbientMIDIPipeline':
        """Create a standard AmbientMIDI pipeline."""
        from .pipeline import AmbientMIDIPipeline
        from .config import get_default_config
        
        if config is None:
            config = get_default_config()
        
        return AmbientMIDIPipeline(config)
    
    @staticmethod
    def create_custom_pipeline(config: AmbientMIDIConfig, 
                             step_overrides: Optional[Dict[str, Type[ProcessingStep]]] = None) -> 'AmbientMIDIPipeline':
        """Create a custom pipeline with specified step overrides."""
        from .pipeline import AmbientMIDIPipeline
        
        # Register any custom steps
        if step_overrides:
            for name, step_class in step_overrides.items():
                ProcessingStepFactory.register_step(name, step_class)
        
        return AmbientMIDIPipeline(config)
    
    @staticmethod
    def create_preset_pipeline(preset: str = 'default', **config_overrides) -> 'AmbientMIDIPipeline':
        """Create a pipeline from a configuration preset."""
        config = ConfigurationFactory.create_config(preset, **config_overrides)
        return PipelineFactory.create_standard_pipeline(config)


# Abstract factory for creating related families of objects
class AmbientMIDIAbstractFactory(ABC):
    """Abstract factory for creating families of related AmbientMIDI objects."""
    
    @abstractmethod
    def create_config(self) -> AmbientMIDIConfig:
        """Create a configuration object."""
        pass
    
    @abstractmethod
    def create_pipeline(self, config: AmbientMIDIConfig) -> 'AmbientMIDIPipeline':
        """Create a pipeline object."""
        pass
    
    @abstractmethod
    def create_audio_processor(self, config: AmbientMIDIConfig):
        """Create an audio processor."""
        pass


class StandardAmbientMIDIFactory(AmbientMIDIAbstractFactory):
    """Standard factory for creating AmbientMIDI components."""
    
    def create_config(self) -> AmbientMIDIConfig:
        """Create a standard configuration."""
        return ConfigurationFactory.create_config('default')
    
    def create_pipeline(self, config: AmbientMIDIConfig) -> 'AmbientMIDIPipeline':
        """Create a standard pipeline."""
        return PipelineFactory.create_standard_pipeline(config)
    
    def create_audio_processor(self, config: AmbientMIDIConfig):
        """Create a standard audio processor."""
        return {
            'spectrogram': AudioProcessorFactory.create_spectrogram_processor(config),
            'onset_detector': AudioProcessorFactory.create_onset_detector(config),
            'loader': AudioProcessorFactory.create_audio_loader(config),
            'recorder': AudioProcessorFactory.create_audio_recorder(config)
        }


class HighQualityAmbientMIDIFactory(AmbientMIDIAbstractFactory):
    """High-quality factory for creating AmbientMIDI components."""
    
    def create_config(self) -> AmbientMIDIConfig:
        """Create a high-quality configuration."""
        return ConfigurationFactory.create_config('high_quality')
    
    def create_pipeline(self, config: AmbientMIDIConfig) -> 'AmbientMIDIPipeline':
        """Create a high-quality pipeline."""
        return PipelineFactory.create_standard_pipeline(config)
    
    def create_audio_processor(self, config: AmbientMIDIConfig):
        """Create a high-quality audio processor."""
        return {
            'spectrogram': AudioProcessorFactory.create_spectrogram_processor(config),
            'onset_detector': AudioProcessorFactory.create_onset_detector(config),
            'loader': AudioProcessorFactory.create_audio_loader(config),
            'recorder': AudioProcessorFactory.create_audio_recorder(config)
        }


class FastAmbientMIDIFactory(AmbientMIDIAbstractFactory):
    """Fast processing factory for creating AmbientMIDI components."""
    
    def create_config(self) -> AmbientMIDIConfig:
        """Create a fast processing configuration."""
        return ConfigurationFactory.create_config('fast')
    
    def create_pipeline(self, config: AmbientMIDIConfig) -> 'AmbientMIDIPipeline':
        """Create a fast processing pipeline."""
        return PipelineFactory.create_standard_pipeline(config)
    
    def create_audio_processor(self, config: AmbientMIDIConfig):
        """Create a fast audio processor."""
        return {
            'spectrogram': AudioProcessorFactory.create_spectrogram_processor(config),
            'onset_detector': AudioProcessorFactory.create_onset_detector(config),
            'loader': AudioProcessorFactory.create_audio_loader(config),
            'recorder': AudioProcessorFactory.create_audio_recorder(config)
        }


# Factory registry for getting factories by name
FACTORY_REGISTRY = {
    'standard': StandardAmbientMIDIFactory,
    'high_quality': HighQualityAmbientMIDIFactory,
    'fast': FastAmbientMIDIFactory,
}


def get_factory(factory_type: str = 'standard') -> AmbientMIDIAbstractFactory:
    """Get a factory by type."""
    if factory_type not in FACTORY_REGISTRY:
        available_types = list(FACTORY_REGISTRY.keys())
        raise ConfigurationError(
            f"Unknown factory type: {factory_type}. "
            f"Available types: {available_types}"
        )
    
    factory_class = FACTORY_REGISTRY[factory_type]
    return factory_class()


def register_factory(name: str, factory_class: Type[AmbientMIDIAbstractFactory]) -> None:
    """Register a new factory type."""
    if not issubclass(factory_class, AmbientMIDIAbstractFactory):
        raise ConfigurationError("Factory must inherit from AmbientMIDIAbstractFactory")
    
    FACTORY_REGISTRY[name] = factory_class
    logger.debug(f"Registered factory: {name}")


# Convenience functions
def create_quick_pipeline(preset: str = 'default', **overrides):
    """Quickly create a pipeline with a preset and overrides."""
    return PipelineFactory.create_preset_pipeline(preset, **overrides)


def create_components(factory_type: str = 'standard'):
    """Create a complete set of AmbientMIDI components."""
    factory = get_factory(factory_type)
    config = factory.create_config()
    pipeline = factory.create_pipeline(config)
    audio_processor = factory.create_audio_processor(config)
    
    return {
        'config': config,
        'pipeline': pipeline,
        'audio_processor': audio_processor,
        'factory': factory
    }