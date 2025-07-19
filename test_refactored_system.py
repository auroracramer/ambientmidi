#!/usr/bin/env python3
"""
Test script for the refactored AmbientMIDI system.

This script tests the basic functionality of the refactored system
without requiring all the audio processing dependencies.
"""

def test_basic_imports():
    """Test basic imports work correctly."""
    print("Testing basic imports...")
    
    try:
        from ambientmidi import get_version_info
        version_info = get_version_info()
        print(f"✅ Version info: {version_info}")
        
        from ambientmidi.config import AmbientMIDIConfig, get_default_config
        config = get_default_config()
        print(f"✅ Default config created: {config.audio.sample_rate}Hz")
        
        from ambientmidi.exceptions import AmbientMIDIError
        print("✅ Exception imports working")
        
        print("✅ Basic imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration_system():
    """Test the configuration system."""
    print("\nTesting configuration system...")
    
    try:
        from ambientmidi.config import AmbientMIDIConfig, AudioConfig
        
        # Test configuration creation
        config = AmbientMIDIConfig()
        print(f"✅ Configuration created")
        
        # Test configuration modification
        config.audio.sample_rate = 44100
        config.audio.record_duration = 30.0
        
        # Test validation
        config.validate_all()
        print("✅ Configuration validation passed")
        
        # Test saving/loading
        import tempfile
        import json
        from pathlib import Path
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.save(f.name)
            temp_path = f.name
        
        # Load it back
        from ambientmidi.config import load_config
        loaded_config = load_config(temp_path)
        print("✅ Configuration save/load working")
        
        # Clean up
        Path(temp_path).unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration system failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_exception_system():
    """Test the exception system."""
    print("\nTesting exception system...")
    
    try:
        from ambientmidi.exceptions import (
            AmbientMIDIError, 
            ConfigurationError, 
            AudioProcessingError,
            ErrorHandler,
            handle_error
        )
        
        # Test basic exception
        try:
            raise AmbientMIDIError("Test error", {"test": "value"})
        except AmbientMIDIError as e:
            print(f"✅ Exception caught: {e}")
            print(f"✅ Exception details: {e.details}")
        
        # Test error handler
        try:
            with ErrorHandler("Test context"):
                raise ValueError("Test error")
        except AmbientMIDIError as e:
            print(f"✅ Error handler working: {e}")
        
        # Test error conversion
        converted_error = handle_error(ValueError("Test"), "Test context")
        print(f"✅ Error conversion working: {converted_error}")
        
        return True
        
    except Exception as e:
        print(f"❌ Exception system failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utility_functions():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    try:
        from ambientmidi.utils import LazyDict
        
        # Test LazyDict
        def test_func(key):
            return f"value_{key}"
        
        lazy_dict = LazyDict(["a", "b", "c"], test_func)
        print(f"✅ LazyDict created with {len(lazy_dict)} keys")
        print(f"✅ LazyDict value: {lazy_dict['a']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Utility functions failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_convenience_functions():
    """Test convenience functions."""
    print("\nTesting convenience functions...")
    
    try:
        from ambientmidi import create_default_config
        
        # Test convenience config creation
        config = create_default_config(
            sample_rate=44100,
            record_duration=30.0,
            samples_per_instrument=15
        )
        
        print(f"✅ Convenience config created: {config.audio.sample_rate}Hz")
        print(f"✅ Record duration: {config.audio.record_duration}s")
        print(f"✅ Samples per instrument: {config.midi.samples_per_instrument}")
        
        return True
        
    except Exception as e:
        print(f"❌ Convenience functions failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧪 Testing Refactored AmbientMIDI System")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_configuration_system,
        test_exception_system,
        test_utility_functions,
        test_convenience_functions
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n📊 Test Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All core tests passed! The refactored system is working correctly.")
        print("Note: Audio processing tests require additional dependencies.")
    else:
        print(f"\n⚠️ {failed} tests failed. Please check the errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)