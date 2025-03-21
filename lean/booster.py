"""
Improved Booster module that enhances performance and optimizes processing.
This version combines the duplicate implementations and improves caching.
"""

import asyncio
import time
import logging
import functools
import hashlib
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Callable, TypeVar, Optional, Union, Tuple

logger = logging.getLogger(__name__)

# Define a type variable for generic function signatures
T = TypeVar('T')

class Booster:
    """
    Enhances performance of document processing with caching and parallel execution.
    
    This improved version:
    1. Combines both process_in_parallel implementations into a single method
    2. Properly handles both async and sync functions
    3. Improves error handling for better resilience
    4. Provides simpler caching implementation
    """
    
    def __init__(self, cache_dir: str = ".cache", max_workers: int = 4, enable_caching: bool = True):
        """
        Initialize the performance booster.
        
        Args:
            cache_dir: Directory to store cache files
            max_workers: Maximum number of parallel workers
            enable_caching: Whether to enable result caching
        """
        self.max_workers = max_workers
        self.enable_caching = enable_caching
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist and caching is enabled
        if self.enable_caching:
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Cache enabled in directory: {self.cache_dir}")
    
    async def process_in_parallel(self, 
                                items: List[Any], 
                                process_func: Callable[[Any], T], 
                                max_concurrency: Optional[int] = None) -> List[T]:
        """
        Process items in parallel with controlled concurrency.
        Works with both synchronous and asynchronous processing functions.
        
        Args:
            items: List of items to process
            process_func: Function to apply to each item (can be sync or async)
            max_concurrency: Maximum concurrent tasks (defaults to self.max_workers)
            
        Returns:
            List of processed results
        """
        if not items:
            return []
        
        # Use provided concurrency limit or fall back to max_workers
        concurrency_limit = max_concurrency or self.max_workers
        logger.info(f"Processing {len(items)} items with concurrency limit of {concurrency_limit}")
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrency_limit)
        
        # Define worker function that respects the semaphore
        async def worker(item, index):
            # Try to get from cache first if enabled
            if self.enable_caching:
                cached_result = self._get_from_cache(item, process_func)
                if cached_result is not None:
                    logger.debug(f"Cache hit for item {index}")
                    return cached_result
            
            # Not in cache, acquire semaphore and process
            async with semaphore:
                try:
                    # Check if the function is already async
                    if asyncio.iscoroutinefunction(process_func):
                        # Process the item with the async function
                        result = await process_func(item)
                    else:
                        # Run sync function in executor
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(None, process_func, item)
                    
                    # Cache result if enabled
                    if self.enable_caching:
                        self._save_to_cache(item, process_func, result)
                    
                    return result
                except Exception as e:
                    logger.error(f"Error processing item {index}: {e}")
                    # Return error information instead of raising
                    return {
                        "error": str(e),
                        "item_index": index
                    }
        
        # Create tasks for all items
        tasks = [worker(item, i) for i, item in enumerate(items)]
        
        # Execute all tasks and wait for completion
        try:
            results = await asyncio.gather(*tasks)
            logger.info(f"Successfully processed {len(results)} items in parallel")
            return results
        except Exception as e:
            logger.error(f"Error in gather operation: {e}")
            # Return partial results if possible
            return []
    
    def _get_cache_key(self, item: Any, func: Callable) -> str:
        """
        Generate a unique cache key for an item and function.
        
        Args:
            item: The item being processed
            func: The processing function
            
        Returns:
            A unique cache key string
        """
        # Get function name or qualified name if available
        func_name = getattr(func, "__qualname__", func.__name__)
        
        # Get module name if available
        if hasattr(func, "__module__") and func.__module__ is not None:
            func_name = f"{func.__module__}.{func_name}"
        
        # Handle different item types
        if isinstance(item, dict):
            # For dictionaries, create a deterministic string representation
            # Sort keys to ensure consistency
            item_str = str(sorted(item.items()))
        elif isinstance(item, str):
            # For strings, use a truncated representation with length info
            item_str = f"len={len(item)}_{item[:50]}..{item[-50:]}" if len(item) > 100 else item
        else:
            # For other types, use string representation
            item_str = str(item)
        
        # Create a hash for the cache key
        key = f"{func_name}_{hashlib.md5(item_str.encode()).hexdigest()}"
        return key
    
    def _get_from_cache(self, item: Any, func: Callable) -> Optional[Any]:
        """
        Get a result from cache if available.
        
        Args:
            item: The item being processed
            func: The processing function
            
        Returns:
            The cached result if available, otherwise None
        """
        if not self.enable_caching:
            return None
            
        try:
            cache_key = self._get_cache_key(item, func)
            cache_file = Path(self.cache_dir) / f"{cache_key}.pkl"
            
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                logger.debug(f"Cache hit for {func.__name__}")
                return result
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        
        return None
    
    def _save_to_cache(self, item: Any, func: Callable, result: Any) -> None:
        """
        Save a result to cache.
        
        Args:
            item: The item being processed
            func: The processing function
            result: The result to cache
        """
        if not self.enable_caching:
            return
            
        try:
            cache_key = self._get_cache_key(item, func)
            cache_file = Path(self.cache_dir) / f"{cache_key}.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            logger.debug(f"Saved result to cache for {func.__name__}")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    async def process_with_retry(self, 
                               func: Callable, 
                               *args, 
                               max_retries: int = 3, 
                               base_delay: float = 2.0, 
                               **kwargs) -> Any:
        """
        Execute a function with exponential backoff for error handling.
        
        Args:
            func: Function to execute (can be sync or async)
            *args: Positional arguments for the function
            max_retries: Maximum number of retries
            base_delay: Initial delay between retries (in seconds)
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function, or raises exception if max_retries reached
        """
        retries = 0
        last_exception = None
        
        while retries <= max_retries:  # Changed from < to <= to include the initial try
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, 
                        functools.partial(func, *args, **kwargs)
                    )
                return result
            except Exception as e:
                last_exception = e
                retries += 1
                
                if retries > max_retries:
                    logger.error(f"Max retries exceeded for function {func.__name__}: {e}")
                    break
                
                delay = base_delay * (2 ** (retries - 1))  # Start with base_delay
                logger.warning(f"Error in {func.__name__}: {e}. Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
        
        # If we've exhausted retries, raise the last exception
        if last_exception:
            raise last_exception
        
        # This should never happen, but just in case
        raise RuntimeError(f"Failed to execute {func.__name__} after {max_retries} retries")