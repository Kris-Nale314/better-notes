"""
Test script for the improved Booster implementation.
"""

import asyncio
import time
import os
import sys
import tempfile
import random
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lean.booster_v2 import Booster

# Test synchronous function
def sync_processor(item):
    """Simple synchronous processing function that squares a number."""
    time.sleep(0.1)  # Simulate processing time
    return item * item

# Test asynchronous function
async def async_processor(item):
    """Simple asynchronous processing function that squares a number."""
    await asyncio.sleep(0.1)  # Simulate processing time
    return item * item

# Test function that sometimes fails
async def flaky_processor(item):
    """Processing function that occasionally fails to test error handling."""
    await asyncio.sleep(0.1)
    if random.random() < 0.3:  # 30% chance of failure
        raise ValueError(f"Random failure on item {item}")
    return item * item

async def test_sync_processing():
    """Test processing with a synchronous function."""
    print("\n--- Testing with synchronous function ---")
    
    # Create a temporary cache directory
    with tempfile.TemporaryDirectory() as cache_dir:
        booster = Booster(cache_dir=cache_dir, max_workers=4, enable_caching=True)
        
        # Process items
        items = list(range(1, 11))
        print(f"Processing {len(items)} items with sync function...")
        
        start_time = time.time()
        results = await booster.process_in_parallel(items, sync_processor)
        elapsed = time.time() - start_time
        
        print(f"Processing completed in {elapsed:.2f} seconds")
        print(f"Results: {results}")
        
        # Test caching - should be much faster the second time
        print("Processing again with cache...")
        start_time = time.time()
        cached_results = await booster.process_in_parallel(items, sync_processor)
        cached_elapsed = time.time() - start_time
        
        print(f"Cached processing completed in {cached_elapsed:.2f} seconds")
        print(f"Same results: {results == cached_results}")
        
        return all(result == item*item for item, result in zip(items, results))

async def test_async_processing():
    """Test processing with an asynchronous function."""
    print("\n--- Testing with asynchronous function ---")
    
    # Create booster without caching
    booster = Booster(enable_caching=False)
    
    # Process items
    items = list(range(1, 11))
    print(f"Processing {len(items)} items with async function...")
    
    start_time = time.time()
    results = await booster.process_in_parallel(items, async_processor)
    elapsed = time.time() - start_time
    
    print(f"Processing completed in {elapsed:.2f} seconds")
    print(f"Results: {results}")
    
    return all(result == item*item for item, result in zip(items, results))

async def test_error_handling():
    """Test error handling with a function that sometimes fails."""
    print("\n--- Testing error handling ---")
    
    booster = Booster(enable_caching=False)
    
    # Process items
    items = list(range(1, 11))
    print(f"Processing {len(items)} items with flaky function...")
    
    results = await booster.process_in_parallel(items, flaky_processor)
    
    # Count successes and failures
    successes = sum(1 for r in results if isinstance(r, int))
    failures = sum(1 for r in results if isinstance(r, dict) and "error" in r)
    
    print(f"Successes: {successes}, Failures: {failures}")
    print(f"Results: {results}")
    
    return successes + failures == len(items)

async def test_retry_mechanism():
    """Test the retry mechanism."""
    print("\n--- Testing retry mechanism ---")
    
    booster = Booster()
    
    # Function that fails twice then succeeds
    failure_count = 0
    
    async def temporary_failure():
        nonlocal failure_count
        failure_count += 1
        if failure_count <= 2:
            raise ValueError(f"Temporary failure {failure_count}")
        return "Success on 3rd try!"
    
    try:
        result = await booster.process_with_retry(
            temporary_failure,
            max_retries=3,
            base_delay=0.1
        )
        print(f"Result after retries: {result}")
        print(f"Total attempts: {failure_count}")
        return result == "Success on 3rd try!"
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

async def run_all_tests():
    """Run all test cases."""
    tests = [
        test_sync_processing,
        test_async_processing,
        test_error_handling,
        test_retry_mechanism
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
            status = "✅ Passed" if result else "❌ Failed"
        except Exception as e:
            results.append(False)
            status = f"❌ Error: {e}"
        
        print(f"{test.__name__}: {status}")
    
    # Overall summary
    all_passed = all(results)
    print(f"\nOverall: {'✅ All tests passed!' if all_passed else '❌ Some tests failed'}")
    return all_passed

if __name__ == "__main__":
    asyncio.run(run_all_tests())