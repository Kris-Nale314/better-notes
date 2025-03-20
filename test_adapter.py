#!/usr/bin/env python
"""
Test script to verify that the OpenAI adapter is working correctly.
Run this script to check if your API key and adapter are configured properly.
"""

import sys
import os
import asyncio

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

# Import the adapter
from lean.async_openai_adapter import AsyncOpenAIAdapter

async def test_adapter():
    """Test the AsyncOpenAIAdapter with a simple prompt."""
    print("\n=== Testing OpenAI Adapter ===")
    
    try:
        # Check if API key is in environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  WARNING: OPENAI_API_KEY environment variable not found.")
            print("   Make sure your .env file contains the API key or set it in your environment.")
            return False
        
        print(f"✓ API key found in environment variables")
        
        # Initialize the adapter
        print("Initializing adapter...")
        adapter = AsyncOpenAIAdapter(model="gpt-3.5-turbo")
        print(f"✓ Adapter initialized with model: {adapter.model}")
        
        # Test synchronous completion
        print("\nTesting synchronous completion...")
        sync_result = adapter.generate_completion("Say hello in a formal way.")
        print(f"✓ Sync result: {sync_result}")
        
        # Test asynchronous completion
        print("\nTesting asynchronous completion...")
        async_result = await adapter.generate_completion_async("Say hello in a casual way.")
        print(f"✓ Async result: {async_result}")
        
        # Test structured output
        print("\nTesting structured output...")
        schema = {
            "type": "object",
            "properties": {
                "greeting": {
                    "type": "string"
                },
                "farewell": {
                    "type": "string"
                }
            }
        }
        
        struct_result = await adapter.generate_completion_with_structured_output(
            "Provide a greeting and a farewell in JSON format.",
            schema
        )
        print(f"✓ Structured result: {struct_result}")
        
        print("\n✅ All tests passed! Your OpenAI adapter is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_adapter())
    if not success:
        sys.exit(1)