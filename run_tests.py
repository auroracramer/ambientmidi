#!/usr/bin/env python3
"""
Comprehensive test runner for AmbientMIDI.

This script runs all unit tests with options for:
- Coverage reporting
- Parallel execution
- Test filtering
- Output formatting
- Performance analysis
"""

import sys
import os
import unittest
import argparse
import time
from pathlib import Path
from io import StringIO
import json

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

# Optional imports for enhanced functionality
try:
    import coverage
    HAS_COVERAGE = True
except ImportError:
    HAS_COVERAGE = False
    print("Warning: coverage.py not installed. Install with: pip install coverage")

try:
    import xmlrunner
    HAS_XMLRUNNER = True
except ImportError:
    HAS_XMLRUNNER = False
    print("Info: xmlrunner not available. Install with: pip install xmlrunner for XML output")


class ColoredTextTestResult(unittest.TextTestResult):
    """Test result class with colored output."""
    
    def __init__(self, stream, descriptions, verbosity, use_colors=True):
        super().__init__(stream, descriptions, verbosity)
        self.use_colors = use_colors and hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        
        # Color codes
        self.colors = {
            'green': '\033[92m',
            'red': '\033[91m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
            'bold': '\033[1m',
            'end': '\033[0m'
        }
    
    def _color(self, text, color):
        """Apply color to text if colors are enabled."""
        if self.use_colors and color in self.colors:
            return f"{self.colors[color]}{text}{self.colors['end']}"
        return text
    
    def addSuccess(self, test):
        super().addSuccess(test)
        if self.showAll:
            self.stream.writeln(self._color("ok", "green"))
        elif self.dots:
            self.stream.write(self._color(".", "green"))
            self.stream.flush()
    
    def addError(self, test, err):
        super().addError(test, err)
        if self.showAll:
            self.stream.writeln(self._color("ERROR", "red"))
        elif self.dots:
            self.stream.write(self._color("E", "red"))
            self.stream.flush()
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.showAll:
            self.stream.writeln(self._color("FAIL", "red"))
        elif self.dots:
            self.stream.write(self._color("F", "red"))
            self.stream.flush()
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.showAll:
            self.stream.writeln(self._color(f"skipped {reason!r}", "yellow"))
        elif self.dots:
            self.stream.write(self._color("s", "yellow"))
            self.stream.flush()


class ColoredTextTestRunner(unittest.TextTestRunner):
    """Test runner with colored output."""
    
    def __init__(self, **kwargs):
        use_colors = kwargs.pop('use_colors', True)
        super().__init__(**kwargs)
        self.use_colors = use_colors
    
    def _makeResult(self):
        return ColoredTextTestResult(
            self.stream, 
            self.descriptions, 
            self.verbosity, 
            use_colors=self.use_colors
        )


class TestStats:
    """Class to track and report test statistics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.test_times = {}
        self.module_times = {}
        
    def record_test_time(self, test_name, duration):
        """Record the execution time of a test."""
        self.test_times[test_name] = duration
        
    def record_module_time(self, module_name, duration):
        """Record the execution time of a test module."""
        self.module_times[module_name] = duration
        
    def get_total_time(self):
        """Get total execution time."""
        return time.time() - self.start_time
        
    def get_slowest_tests(self, n=10):
        """Get the n slowest tests."""
        return sorted(self.test_times.items(), key=lambda x: x[1], reverse=True)[:n]
        
    def get_slowest_modules(self, n=5):
        """Get the n slowest modules."""
        return sorted(self.module_times.items(), key=lambda x: x[1], reverse=True)[:n]


def discover_tests(test_dir="tests", pattern="test_*.py"):
    """Discover test modules."""
    loader = unittest.TestLoader()
    
    # If test_dir doesn't exist, look for tests in the current directory
    if not Path(test_dir).exists():
        test_dir = "."
        pattern = "test_*.py"
    
    try:
        suite = loader.discover(test_dir, pattern=pattern)
        return suite
    except Exception as e:
        print(f"Error discovering tests: {e}")
        return unittest.TestSuite()


def run_tests_with_coverage(test_suite, verbosity=2, use_colors=True):
    """Run tests with coverage reporting."""
    if not HAS_COVERAGE:
        print("Coverage reporting not available. Running tests without coverage.")
        return run_tests_basic(test_suite, verbosity, use_colors)
    
    # Start coverage
    cov = coverage.Coverage(source=['ambientmidi'])
    cov.start()
    
    try:
        # Run tests
        result = run_tests_basic(test_suite, verbosity, use_colors)
        
        # Stop coverage and generate report
        cov.stop()
        cov.save()
        
        print("\n" + "="*70)
        print("COVERAGE REPORT")
        print("="*70)
        
        # Console report
        cov.report(show_missing=True)
        
        # HTML report
        html_dir = Path("htmlcov")
        if html_dir.exists():
            import shutil
            shutil.rmtree(html_dir)
        
        cov.html_report(directory=str(html_dir))
        print(f"\nHTML coverage report generated in: {html_dir}")
        
        # JSON report for CI/CD
        cov.json_report(outfile="coverage.json")
        
        return result
        
    except Exception as e:
        print(f"Error during coverage analysis: {e}")
        return run_tests_basic(test_suite, verbosity, use_colors)


def run_tests_basic(test_suite, verbosity=2, use_colors=True):
    """Run tests without coverage."""
    runner = ColoredTextTestRunner(
        verbosity=verbosity,
        use_colors=use_colors,
        buffer=True,
        failfast=False
    )
    
    start_time = time.time()
    result = runner.run(test_suite)
    end_time = time.time()
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    
    return result


def run_tests_xml(test_suite, output_dir="test-reports"):
    """Run tests with XML output for CI/CD."""
    if not HAS_XMLRUNNER:
        print("XML runner not available. Running basic tests.")
        return run_tests_basic(test_suite)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Run tests with XML output
    runner = xmlrunner.XMLTestRunner(
        output=str(output_path),
        verbosity=2,
        buffer=True
    )
    
    result = runner.run(test_suite)
    print(f"\nXML test reports generated in: {output_path}")
    
    return result


def filter_tests(test_suite, pattern):
    """Filter tests by pattern."""
    filtered_suite = unittest.TestSuite()
    
    def add_matching_tests(suite_or_test):
        if isinstance(suite_or_test, unittest.TestSuite):
            for test in suite_or_test:
                add_matching_tests(test)
        else:
            test_name = str(suite_or_test)
            if pattern.lower() in test_name.lower():
                filtered_suite.addTest(suite_or_test)
    
    add_matching_tests(test_suite)
    return filtered_suite


def print_test_info(test_suite):
    """Print information about discovered tests."""
    test_count = test_suite.countTestCases()
    
    print(f"Discovered {test_count} tests")
    
    # Group tests by module
    modules = {}
    
    def collect_tests(suite_or_test):
        if isinstance(suite_or_test, unittest.TestSuite):
            for test in suite_or_test:
                collect_tests(test)
        else:
            module_name = suite_or_test.__class__.__module__
            if module_name not in modules:
                modules[module_name] = 0
            modules[module_name] += 1
    
    collect_tests(test_suite)
    
    print("\nTests by module:")
    for module, count in sorted(modules.items()):
        print(f"  {module}: {count} tests")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Run AmbientMIDI unit tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Run all tests
  %(prog)s --coverage               # Run with coverage
  %(prog)s --filter config          # Run only config tests
  %(prog)s --xml                    # Generate XML reports
  %(prog)s -v 1                     # Quiet output
  %(prog)s --no-colors              # Disable colored output
        """
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Run tests with coverage reporting"
    )
    
    parser.add_argument(
        "--xml",
        action="store_true",
        help="Generate XML test reports"
    )
    
    parser.add_argument(
        "--filter", "-f",
        type=str,
        help="Filter tests by pattern"
    )
    
    parser.add_argument(
        "--verbosity", "-v",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="Test output verbosity (0=quiet, 1=normal, 2=verbose)"
    )
    
    parser.add_argument(
        "--no-colors",
        action="store_true",
        help="Disable colored output"
    )
    
    parser.add_argument(
        "--test-dir",
        default="tests",
        help="Directory containing tests (default: tests)"
    )
    
    parser.add_argument(
        "--pattern",
        default="test_*.py",
        help="Pattern for test file discovery (default: test_*.py)"
    )
    
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List discovered tests and exit"
    )
    
    args = parser.parse_args()
    
    # Discover tests
    print("Discovering tests...")
    test_suite = discover_tests(args.test_dir, args.pattern)
    
    if test_suite.countTestCases() == 0:
        print("No tests found!")
        return 1
    
    # Filter tests if requested
    if args.filter:
        print(f"Filtering tests by pattern: {args.filter}")
        test_suite = filter_tests(test_suite, args.filter)
        if test_suite.countTestCases() == 0:
            print("No tests match the filter pattern!")
            return 1
    
    # Print test information
    print_test_info(test_suite)
    
    if args.list_tests:
        return 0
    
    print("\nRunning tests...")
    print("="*70)
    
    # Run tests
    use_colors = not args.no_colors
    
    if args.xml:
        result = run_tests_xml(test_suite)
    elif args.coverage:
        result = run_tests_with_coverage(test_suite, args.verbosity, use_colors)
    else:
        result = run_tests_basic(test_suite, args.verbosity, use_colors)
    
    # Return appropriate exit code
    if result.failures or result.errors:
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())