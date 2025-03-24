"""
Universal LLM Adapter - Provides a consistent interface across different LLM clients.
Works with both personal AsyncOpenAIAdapter and work CustomLLMAdapter.
"""

import os
import logging
import asyncio
import json
from typing import Optional, Dict, Any, Union, Callable

logger = logging.getLogger(__name__)

class UniversalLLMAdapter:
    """
    A universal adapter that works with different LLM implementations.
    Supports both personal (AsyncOpenAIAdapter) and work environments.
    """
    
    def __init__(
        self, 
        llm_client=None,
        api_key=None, 
        model="gpt-3.5-turbo", 
        temperature=0.2
    ):
        """Initialize with either an existing client or credentials."""
        self.model = model
        self.temperature = temperature
        self.client = llm_client
        
        # Only try to initialize if no client provided
        if not self.client:
            self._initialize_client(api_key, model, temperature)
    
    def _initialize_client(self, api_key, model, temperature):
        """Try different client initializations based on available imports."""
        # Try work adapter first (assuming it exists in your environment)
        try:
            # Import your work's CustomLLMAdapter if available
            # This is a placeholder - replace with your actual work adapter import
            from work_lib.custom_llm_adapter import CustomLLMAdapter
            logger.info("Using work CustomLLMAdapter")
            self.client = CustomLLMAdapter(
                api_key=api_key,
                model=model,
                temperature=temperature
            )
            return
        except ImportError:
            logger.info("Work LLM adapter not found, trying personal adapter")
        
        # Try personal adapter
        try:
            # Import your personal AsyncOpenAIAdapter
            from lean.async_openai_adapter import AsyncOpenAIAdapter
            logger.info("Using personal AsyncOpenAIAdapter")
            self.client = AsyncOpenAIAdapter(
                api_key=api_key,
                model=model,
                temperature=temperature
            )
            return
        except ImportError:
            logger.warning("Personal LLM adapter not found")
        
        # If both fail, try direct OpenAI
        try:
            from openai import OpenAI, AsyncOpenAI
            logger.info("Using direct OpenAI client")
            self.direct_client = OpenAI(api_key=api_key)
            self.direct_async_client = AsyncOpenAI(api_key=api_key)
            return
        except ImportError:
            logger.error("No LLM client implementation available")
            raise ImportError("Could not initialize any LLM client")
    
    async def generate_completion_async(self, prompt: str, max_retries: int = 2) -> str:
        """
        Generate completion asynchronously using whatever client is available.
        Adapts to different client interfaces.
        """
        if not self.client and not hasattr(self, 'direct_async_client'):
            raise ValueError("No LLM client initialized")
        
        # Use direct OpenAI client if available and no other client is set
        if not self.client and hasattr(self, 'direct_async_client'):
            # Use direct OpenAI client
            for retry in range(max_retries + 1):
                try:
                    response = await self.direct_async_client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    if retry >= max_retries:
                        logger.error(f"Max retries exceeded: {e}")
                        raise
                    wait_time = 2 ** retry
                    logger.warning(f"API error, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
        
        # Try different methods based on what's available in the client
        if hasattr(self.client, 'generate_completion_async'):
            return await self.client.generate_completion_async(prompt, max_retries)
        elif hasattr(self.client, 'generate_completion'):
            # Handle both sync and async versions
            if asyncio.iscoroutinefunction(self.client.generate_completion):
                return await self.client.generate_completion(prompt, max_retries)
            else:
                # Run sync method in executor
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, 
                    lambda: self.client.generate_completion(prompt, max_retries)
                )
        elif hasattr(self.client, 'completion'):
            # Try alternative method name
            if asyncio.iscoroutinefunction(self.client.completion):
                return await self.client.completion(prompt)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: self.client.completion(prompt))
        
        # Fall back to assuming client is callable
        if callable(self.client):
            if asyncio.iscoroutinefunction(self.client):
                return await self.client(prompt)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: self.client(prompt))
        
        raise NotImplementedError("LLM client doesn't have a recognized completion method")
    
    def generate_completion(self, prompt: str, max_retries: int = 2) -> str:
        """
        Generate completion synchronously by running the async method in an event loop.
        """
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in a running loop, need to use a new one
                loop = asyncio.new_event_loop()
        except RuntimeError:
            # No event loop in thread, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            # Run the async method in the event loop
            return loop.run_until_complete(self.generate_completion_async(prompt, max_retries))
        except Exception as e:
            logger.error(f"Error in generate_completion: {e}")
            raise
        finally:
            # Close the loop if we created a new one
            if not asyncio.get_event_loop().is_running():
                loop.close()
    
    async def generate_completion_with_structured_output(self, 
                                                       prompt: str,
                                                       output_schema: Dict[str, Any],
                                                       max_retries: int = 3) -> Dict[str, Any]:
        """
        Generate completion with structured output in JSON format.
        Adapts to different client interfaces.
        """
        # Forward to client method if available
        if hasattr(self.client, 'generate_completion_with_structured_output'):
            if asyncio.iscoroutinefunction(self.client.generate_completion_with_structured_output):
                return await self.client.generate_completion_with_structured_output(
                    prompt, output_schema, max_retries
                )
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, 
                    lambda: self.client.generate_completion_with_structured_output(
                        prompt, output_schema, max_retries
                    )
                )
        
        # Fall back to standard completion + JSON parsing
        for retry in range(max_retries):
            try:
                response = await self.generate_completion_async(
                    prompt + "\n\nReturn your response in valid JSON format.",
                    max_retries=1
                )
                
                # Try to extract JSON
                try:
                    # Look for JSON block in response
                    json_match = response
                    if "```json" in response:
                        json_match = response.split("```json")[1].split("```")[0].strip()
                    elif "```" in response:
                        json_match = response.split("```")[1].split("```")[0].strip()
                    
                    # Parse the JSON
                    result = json.loads(json_match)
                    return result
                except json.JSONDecodeError as e:
                    if retry < max_retries - 1:
                        logger.warning(f"JSON parsing error, retrying: {e}")
                        await asyncio.sleep(2 ** retry)
                    else:
                        logger.error(f"Failed to parse JSON after {max_retries} attempts")
                        # Return the text response as fallback
                        return {"text": response, "error": "JSON parsing failed"}
                        
            except Exception as e:
                if retry < max_retries - 1:
                    logger.warning(f"Error in structured output, retrying: {e}")
                    await asyncio.sleep(2 ** retry)
                else:
                    logger.error(f"Failed to generate structured output: {e}")
                    raise