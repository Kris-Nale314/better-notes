# lean/async_openai_adapter.py
import os
import logging
import time
import asyncio
import json
from typing import Optional, Dict, Any, List

from openai import OpenAI
from openai import AsyncOpenAI
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
# Use find_dotenv to locate the .env file in parent directories if needed
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)
    print(f"Loaded environment variables from {dotenv_path}")
else:
    print("No .env file found")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AsyncOpenAIAdapter:
    """OpenAI adapter with async support."""
    
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None, 
                temperature: float = 0.2):
        """
        Initialize the OpenAI adapter.
        
        Args:
            model: Model name to use
            api_key: OpenAI API key (defaults to environment variable)
            temperature: Temperature for generation
        """
        self.model = model
        self.temperature = temperature
        
        # Set API key - check multiple possible environment variable names
        api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
        
        if not api_key:
            available_vars = [key for key in os.environ.keys() if "OPENAI" in key.upper()]
            msg = "OpenAI API key not provided and not found in environment variables."
            if available_vars:
                msg += f" Found these OpenAI-related variables: {available_vars}"
            logger.error(msg)
            raise ValueError(msg)
        
        # Log successful API key setup (without revealing the key)
        api_key_preview = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
        logger.info(f"Using OpenAI API key: {api_key_preview}")
        
        # Initialize both sync and async clients
        try:
            self.client = OpenAI(api_key=api_key)
            self.async_client = AsyncOpenAI(api_key=api_key)
            logger.info(f"OpenAI clients initialized with model: {model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI clients: {e}")
            raise
    
    def generate_completion(self, prompt: str, max_retries: int = 2) -> str:
        """
        Generate a completion from the OpenAI API (synchronous).
        
        Args:
            prompt: Prompt to complete
            max_retries: Maximum number of retries on error
            
        Returns:
            Generated completion text
        """
        logger.info(f"Generating completion with model: {self.model}, length: {len(prompt)} chars")
        retries = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                
                result = response.choices[0].message.content
                logger.info(f"Completion generated successfully, length: {len(result)} chars")
                return result
                
            except Exception as e:
                retries += 1
                logger.warning(f"API error ({retries}/{max_retries}): {e}")
                if retries > max_retries:
                    logger.error(f"Max retries exceeded: {e}")
                    raise
                
                # Simple exponential backoff
                wait_time = 2 ** retries
                logger.warning(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
    
    async def generate_completion_async(self, prompt: str, max_retries: int = 2) -> str:
        """
        Generate a completion from the OpenAI API (asynchronous).
        
        Args:
            prompt: Prompt to complete
            max_retries: Maximum number of retries on error
            
        Returns:
            Generated completion text
        """
        logger.info(f"Generating async completion with model: {self.model}, length: {len(prompt)} chars")
        retries = 0
        while True:
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                
                result = response.choices[0].message.content
                logger.info(f"Async completion generated successfully, length: {len(result)} chars")
                return result
                
            except Exception as e:
                retries += 1
                logger.warning(f"API error ({retries}/{max_retries}): {e}")
                if retries > max_retries:
                    logger.error(f"Max retries exceeded: {e}")
                    raise

                # Simple exponential backoff
                wait_time = 2 ** retries
                logger.warning(f"Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

    async def generate_completion_with_structured_output(self,
                                                        prompt: str,
                                                        output_schema: Dict[str, Any],
                                                        max_retries: int = 3) -> Dict[str, Any]:
        """
        Generate completion with structured output in JSON format.

        Args:
            prompt: Prompt for generation
            output_schema: Expected JSON output schema
            max_retries: Maximum retries

        Returns:
            Dictionary with parsed JSON output, or None if parsing fails
        """
        logger.info(f"Generating structured output with model: {self.model}, schema: {output_schema.get('type', 'unknown')}")
        retries = 0
        while retries < max_retries:
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )
                response_text = response.choices[0].message.content
                
                # Try to parse as JSON
                try:
                    parsed_output = json.loads(response_text)
                    logger.info(f"Structured output generated successfully")
                    return parsed_output
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parsing error: {e}. Response: {response_text[:100]}...")
                    # Retry
                    retries += 1
                    wait_time = 2 ** retries
                    logger.warning(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue

            except Exception as e:
                logger.error(f"API call error: {str(e)}")
                retries += 1
                if retries >= max_retries:
                    logger.error("Max retries exceeded. Returning None.")
                    return None
                wait_time = 2 ** retries
                logger.warning(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
        
        logger.error(f"Failed to get structured output after {max_retries} retries.")
        return None