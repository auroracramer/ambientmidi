"""
Unit tests for the AmbientMIDI utilities module.

Tests cover:
- LazyDict implementation and behavior
- NpEncoder JSON serialization
- Utility functions (qtile)
- Edge cases and error handling
"""

import unittest
import json
import numpy as np
from collections.abc import Mapping

from ambientmidi.utils import LazyDict, qtile, NpEncoder


class TestLazyDict(unittest.TestCase):
    """Test LazyDict class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.keys = ['a', 'b', 'c', 'd']
        self.values = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        
        def lookup_func(key):
            return self.values[key]
        
        self.lazy_dict = LazyDict(self.keys, lookup_func)
    
    def test_is_mapping(self):
        """Test that LazyDict is a proper Mapping."""
        self.assertIsInstance(self.lazy_dict, Mapping)
    
    def test_getitem_valid_key(self):
        """Test getting item with valid key."""
        self.assertEqual(self.lazy_dict['a'], 1)
        self.assertEqual(self.lazy_dict['b'], 2)
        self.assertEqual(self.lazy_dict['c'], 3)
        self.assertEqual(self.lazy_dict['d'], 4)
    
    def test_getitem_invalid_key(self):
        """Test getting item with invalid key."""
        with self.assertRaises(KeyError):
            _ = self.lazy_dict['invalid_key']
        
        with self.assertRaises(KeyError):
            _ = self.lazy_dict['e']
    
    def test_contains(self):
        """Test __contains__ method (in operator)."""
        self.assertIn('a', self.lazy_dict)
        self.assertIn('b', self.lazy_dict)
        self.assertNotIn('invalid_key', self.lazy_dict)
        self.assertNotIn('e', self.lazy_dict)
    
    def test_iter(self):
        """Test iteration over keys."""
        keys = list(self.lazy_dict)
        self.assertEqual(set(keys), set(self.keys))
    
    def test_len(self):
        """Test length of LazyDict."""
        self.assertEqual(len(self.lazy_dict), len(self.keys))
    
    def test_keys(self):
        """Test keys() method."""
        keys = list(self.lazy_dict.keys())
        self.assertEqual(set(keys), set(self.keys))
    
    def test_values(self):
        """Test values() method."""
        values = list(self.lazy_dict.values())
        expected_values = [self.values[key] for key in self.keys]
        self.assertEqual(set(values), set(expected_values))
    
    def test_items(self):
        """Test items() method."""
        items = list(self.lazy_dict.items())
        expected_items = [(key, self.values[key]) for key in self.keys]
        self.assertEqual(set(items), set(expected_items))
    
    def test_lazy_evaluation(self):
        """Test that function is called only when needed."""
        call_count = 0
        
        def counting_func(key):
            nonlocal call_count
            call_count += 1
            return self.values[key]
        
        lazy_dict = LazyDict(self.keys, counting_func)
        
        # No calls yet
        self.assertEqual(call_count, 0)
        
        # Access one key
        _ = lazy_dict['a']
        self.assertEqual(call_count, 1)
        
        # Access same key again - should call function again (not cached)
        _ = lazy_dict['a']
        self.assertEqual(call_count, 2)
        
        # Access different key
        _ = lazy_dict['b']
        self.assertEqual(call_count, 3)
    
    def test_function_exception(self):
        """Test behavior when function raises exception."""
        def failing_func(key):
            if key == 'fail':
                raise ValueError("Test error")
            return self.values[key]
        
        lazy_dict = LazyDict(self.keys + ['fail'], failing_func)
        
        # Normal keys should work
        self.assertEqual(lazy_dict['a'], 1)
        
        # Failing key should propagate exception
        with self.assertRaises(ValueError) as cm:
            _ = lazy_dict['fail']
        self.assertIn("Test error", str(cm.exception))
    
    def test_empty_lazy_dict(self):
        """Test LazyDict with no keys."""
        def dummy_func(key):
            return None
        
        empty_dict = LazyDict([], dummy_func)
        
        self.assertEqual(len(empty_dict), 0)
        self.assertEqual(list(empty_dict), [])
        
        with self.assertRaises(KeyError):
            _ = empty_dict['any_key']


class TestQtile(unittest.TestCase):
    """Test qtile (percentile) function."""
    
    def test_qtile_basic(self):
        """Test basic qtile functionality."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Test default 45th percentile
        result = qtile(data)
        expected = np.percentile(data, 45)
        self.assertEqual(result, expected)
        
        # Test specific percentiles
        result_25 = qtile(data, 25)
        expected_25 = np.percentile(data, 25)
        self.assertEqual(result_25, expected_25)
        
        result_75 = qtile(data, 75)
        expected_75 = np.percentile(data, 75)
        self.assertEqual(result_75, expected_75)
    
    def test_qtile_edge_cases(self):
        """Test qtile with edge cases."""
        # Single element
        single = np.array([5])
        self.assertEqual(qtile(single, 50), 5)
        
        # Two elements
        two = np.array([1, 9])
        result = qtile(two, 50)
        expected = np.percentile(two, 50)
        self.assertEqual(result, expected)
        
        # All same values
        same = np.array([3, 3, 3, 3, 3])
        self.assertEqual(qtile(same, 50), 3)
    
    def test_qtile_with_args_kwargs(self):
        """Test qtile with additional args and kwargs."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Test with axis argument
        result = qtile(data, 50, axis=0)
        expected = np.percentile(data, 50, axis=0)
        np.testing.assert_array_equal(result, expected)
        
        result = qtile(data, 50, axis=1)
        expected = np.percentile(data, 50, axis=1)
        np.testing.assert_array_equal(result, expected)
        
        # Test with method keyword argument (if numpy version supports it)
        try:
            result = qtile(data, 50, method='linear')
            expected = np.percentile(data, 50, method='linear')
            np.testing.assert_array_equal(result, expected)
        except TypeError:
            # Older numpy versions don't support method parameter
            pass
    
    def test_qtile_float_input(self):
        """Test qtile with float input data."""
        data = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        
        result = qtile(data, 50)
        expected = np.percentile(data, 50)
        self.assertAlmostEqual(result, expected, places=10)
    
    def test_qtile_percentile_bounds(self):
        """Test qtile with boundary percentile values."""
        data = np.array([1, 2, 3, 4, 5])
        
        # 0th percentile (minimum)
        result_0 = qtile(data, 0)
        self.assertEqual(result_0, 1)
        
        # 100th percentile (maximum)
        result_100 = qtile(data, 100)
        self.assertEqual(result_100, 5)


class TestNpEncoder(unittest.TestCase):
    """Test NpEncoder JSON encoder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = NpEncoder()
    
    def test_encode_numpy_integer(self):
        """Test encoding numpy integer types."""
        # Test various numpy integer types
        int_types = [np.int8, np.int16, np.int32, np.int64, 
                     np.uint8, np.uint16, np.uint32, np.uint64]
        
        for int_type in int_types:
            value = int_type(42)
            encoded = self.encoder.default(value)
            self.assertEqual(encoded, 42)
            self.assertIsInstance(encoded, int)
    
    def test_encode_numpy_floating(self):
        """Test encoding numpy floating types."""
        # Test various numpy floating types
        float_types = [np.float16, np.float32, np.float64]
        
        for float_type in float_types:
            value = float_type(3.14)
            encoded = self.encoder.default(value)
            self.assertAlmostEqual(encoded, 3.14, places=5)
            self.assertIsInstance(encoded, float)
    
    def test_encode_numpy_array(self):
        """Test encoding numpy arrays."""
        # 1D array
        array_1d = np.array([1, 2, 3, 4, 5])
        encoded = self.encoder.default(array_1d)
        self.assertEqual(encoded, [1, 2, 3, 4, 5])
        self.assertIsInstance(encoded, list)
        
        # 2D array
        array_2d = np.array([[1, 2], [3, 4]])
        encoded = self.encoder.default(array_2d)
        self.assertEqual(encoded, [[1, 2], [3, 4]])
        self.assertIsInstance(encoded, list)
        
        # Float array
        array_float = np.array([1.1, 2.2, 3.3])
        encoded = self.encoder.default(array_float)
        self.assertEqual(len(encoded), 3)
        self.assertAlmostEqual(encoded[0], 1.1, places=5)
    
    def test_encode_mixed_numpy_array(self):
        """Test encoding arrays with different numpy types."""
        # Array with mixed types
        array = np.array([np.int32(1), np.float64(2.5), np.int64(3)])
        encoded = self.encoder.default(array)
        
        self.assertEqual(len(encoded), 3)
        self.assertEqual(encoded[0], 1)
        self.assertAlmostEqual(encoded[1], 2.5, places=5)
        self.assertEqual(encoded[2], 3)
    
    def test_encode_unsupported_type(self):
        """Test encoding unsupported types raises TypeError."""
        class CustomClass:
            pass
        
        custom_obj = CustomClass()
        
        with self.assertRaises(TypeError):
            self.encoder.default(custom_obj)
    
    def test_json_dumps_with_encoder(self):
        """Test using NpEncoder with json.dumps."""
        data = {
            'int': np.int32(42),
            'float': np.float64(3.14),
            'array': np.array([1, 2, 3]),
            'nested': {
                'bool': True,
                'numpy_val': np.int16(100)
            }
        }
        
        # Should not raise exception
        json_str = json.dumps(data, cls=NpEncoder)
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        
        self.assertEqual(parsed['int'], 42)
        self.assertAlmostEqual(parsed['float'], 3.14, places=5)
        self.assertEqual(parsed['array'], [1, 2, 3])
        self.assertEqual(parsed['nested']['numpy_val'], 100)
    
    def test_json_dumps_complex_structure(self):
        """Test encoding complex nested structures."""
        data = {
            'metadata': {
                'sample_rate': np.int32(16000),
                'duration': np.float64(30.5),
                'channels': np.uint8(2)
            },
            'audio_features': {
                'mfcc': np.random.random(40).astype(np.float32),
                'pitch': np.float64(440.0),
                'energy': np.array([0.1, 0.2, 0.3, 0.4])
            },
            'processing_time': np.float64(1.234)
        }
        
        # Should encode successfully
        json_str = json.dumps(data, cls=NpEncoder)
        
        # Should parse back correctly
        parsed = json.loads(json_str)
        
        self.assertEqual(parsed['metadata']['sample_rate'], 16000)
        self.assertAlmostEqual(parsed['metadata']['duration'], 30.5, places=5)
        self.assertEqual(len(parsed['audio_features']['mfcc']), 40)
        self.assertEqual(parsed['audio_features']['energy'], [0.1, 0.2, 0.3, 0.4])
    
    def test_encoder_preserves_standard_types(self):
        """Test that encoder doesn't interfere with standard Python types."""
        data = {
            'string': 'hello',
            'int': 42,
            'float': 3.14,
            'bool': True,
            'list': [1, 2, 3],
            'dict': {'key': 'value'},
            'none': None
        }
        
        # Should encode successfully without modification
        json_str = json.dumps(data, cls=NpEncoder)
        parsed = json.loads(json_str)
        
        self.assertEqual(parsed, data)


class TestUtilsIntegration(unittest.TestCase):
    """Integration tests for utilities working together."""
    
    def test_lazy_dict_with_numpy_encoder(self):
        """Test LazyDict values can be encoded with NpEncoder."""
        def numpy_value_func(key):
            if key == 'array':
                return np.array([1, 2, 3])
            elif key == 'int':
                return np.int32(42)
            elif key == 'float':
                return np.float64(3.14)
            else:
                return f"value_{key}"
        
        keys = ['array', 'int', 'float', 'string']
        lazy_dict = LazyDict(keys, numpy_value_func)
        
        # Convert to regular dict for JSON encoding
        regular_dict = {key: lazy_dict[key] for key in keys}
        
        # Should encode successfully
        json_str = json.dumps(regular_dict, cls=NpEncoder)
        parsed = json.loads(json_str)
        
        self.assertEqual(parsed['array'], [1, 2, 3])
        self.assertEqual(parsed['int'], 42)
        self.assertAlmostEqual(parsed['float'], 3.14, places=5)
        self.assertEqual(parsed['string'], 'value_string')
    
    def test_qtile_with_lazy_dict_values(self):
        """Test using qtile on values from LazyDict."""
        def array_func(key):
            if key == 'data1':
                return np.array([1, 2, 3, 4, 5])
            elif key == 'data2':
                return np.array([10, 20, 30, 40, 50])
            else:
                return np.array([0])
        
        keys = ['data1', 'data2', 'empty']
        lazy_dict = LazyDict(keys, array_func)
        
        # Use qtile on lazy dict values
        percentile_1 = qtile(lazy_dict['data1'], 50)
        percentile_2 = qtile(lazy_dict['data2'], 50)
        
        self.assertEqual(percentile_1, 3.0)  # median of [1,2,3,4,5]
        self.assertEqual(percentile_2, 30.0)  # median of [10,20,30,40,50]


if __name__ == '__main__':
    unittest.main()