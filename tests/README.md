# AmbientMIDI Test Suite

This directory contains comprehensive unit tests for the AmbientMIDI system.

## Test Organization

The test suite is organized into modules that mirror the main codebase structure:

- **`test_config.py`** - Configuration system tests including validation, serialization, and presets
- **`test_audio.py`** - Audio processing tests with mocked external dependencies (PyAudio, SoundFile)
- **`test_utils.py`** - Utility function tests including LazyDict, JSON encoding, and helper functions
- **`test_exceptions.py`** - Exception handling tests including error hierarchy and context management
- **`test_pipeline.py`** - Pipeline orchestration tests with mocked processing steps
- **`test_factories.py`** - Factory pattern tests for component creation and management
- **`__init__.py`** - Test fixtures, utilities, and shared test infrastructure

## Running Tests

### Quick Start

```bash
# Run all tests with coverage
python run_tests.py --coverage

# Run specific test modules
python run_tests.py --filter config
python run_tests.py --filter audio

# Run with different verbosity levels
python run_tests.py -v 0  # Quiet
python run_tests.py -v 1  # Normal
python run_tests.py -v 2  # Verbose (default)
```

### Using Make

```bash
# Run all tests with coverage
make test

# Run quick tests only
make test-quick

# Run with XML output for CI
make test-xml

# Run tests in watch mode
make test-watch
```

### Using pytest directly

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=ambientmidi --cov-report=html

# Run specific test file
pytest tests/test_config.py -v

# Run specific test class
pytest tests/test_config.py::TestAudioConfig -v

# Run specific test method
pytest tests/test_config.py::TestAudioConfig::test_default_creation -v
```

## Test Features

### Mocking Strategy

The tests use extensive mocking to isolate components and avoid dependencies on:
- External libraries (PyAudio, librosa, etc.)
- File system operations
- Network calls
- Hardware dependencies

### Test Fixtures

The `TestFixtures` class in `__init__.py` provides common test data:
- Mock audio arrays
- Mock MIDI data structures
- Mock spectrograms
- Temporary directory management

### Coverage

Tests aim for high coverage across:
- Normal execution paths
- Error conditions and edge cases
- Configuration validation
- Input/output handling
- Exception propagation

## Test Categories

### Unit Tests
- Test individual functions and classes in isolation
- Use mocking for external dependencies
- Focus on specific functionality and edge cases

### Integration Tests
- Test interaction between components
- Verify data flow through processing pipeline
- Test configuration impact on component behavior

### Validation Tests
- Test input validation and sanitization
- Test configuration parameter bounds
- Test error messages and exception types

## Writing New Tests

When adding new tests, follow these guidelines:

1. **Use descriptive test names** that explain what is being tested
2. **Follow the AAA pattern**: Arrange, Act, Assert
3. **Mock external dependencies** to keep tests isolated and fast
4. **Test both success and failure cases**
5. **Use appropriate test fixtures** from the TestFixtures class
6. **Include docstrings** explaining the test purpose

### Example Test Structure

```python
class TestNewFeature(unittest.TestCase):
    """Test NewFeature class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fixtures = TestFixtures()
        self.config = AmbientMIDIConfig()
    
    def test_successful_operation(self):
        """Test that the feature works correctly with valid input."""
        # Arrange
        input_data = self.fixtures.create_mock_data()
        
        # Act
        result = feature.process(input_data)
        
        # Assert
        self.assertIsNotNone(result)
        self.assertEqual(result.status, 'success')
    
    def test_error_handling(self):
        """Test that the feature handles errors appropriately."""
        # Arrange
        invalid_input = None
        
        # Act & Assert
        with self.assertRaises(InvalidInputError):
            feature.process(invalid_input)
```

## Dependencies

The test suite requires these additional packages (install with `pip install -r requirements-dev.txt`):

- `pytest` - Test runner and framework
- `pytest-cov` - Coverage reporting
- `pytest-mock` - Advanced mocking capabilities
- `coverage` - Code coverage analysis
- `xmlrunner` - XML test reports for CI

## Continuous Integration

The test suite is designed to work with CI/CD systems:

- **XML Reports**: Use `--xml` flag for JUnit-compatible reports
- **Coverage Reports**: Multiple formats (HTML, XML, JSON) supported
- **Exit Codes**: Proper exit codes for CI success/failure detection
- **Parallel Execution**: Support for parallel test execution

## Performance

Tests are designed to be fast and reliable:

- External dependencies are mocked
- Temporary files are properly cleaned up
- Heavy computations are avoided in favor of mock data
- Tests can run in parallel safely