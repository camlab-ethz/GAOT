#!/usr/bin/env python3
"""
Test runner for GAOT unit tests
"""
import sys
import os
import subprocess

def run_neighbor_search_tests():
    """Run neighbor search tests"""
    test_file = "test/model/layers/utils/test_neighbor_search.py"
    
    if not os.path.exists(test_file):
        print(f"Test file {test_file} not found!")
        return False
    
    try:
        # Run tests with pytest
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            test_file, 
            "-v",  # verbose output
            "--tb=short",  # shorter traceback format
        ], capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

def run_all_tests():
    """Run all available tests"""
    print("Running neighbor search tests...")
    success = run_neighbor_search_tests()
    
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    return success

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)