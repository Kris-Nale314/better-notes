# universal_llm_adapter.py
import os
import logging
import json
import asyncio
from typing import Optional, Dict, Any, Union
from openai import OpenAI, AsyncOpenAI

logger = logging.getLogger(__name__)

class LLMAdapter:
    """
    Simplified LLM adapter with consistent interface for all components.
    Removes complex fallback mechanisms while maintaining core functionality.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: str = "gpt-3.5-turbo", 
        temperature: float = 0.2,
        llm_client = None
    ):
        """Initialize the LLM adapter with API key or existing client."""
        self.model = model
        self.temperature = temperature
        
        # Either use provided client or initialize our own
        if llm_client is not None:
            self.client = llm_client
            logger.info(f"Using provided LLM client")
        else:
            # Get API key from parameter or environment
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("No API key provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            
            # Initialize OpenAI clients
            try:
                self.client = OpenAI(api_key=self.api_key)
                self.async_client = AsyncOpenAI(api_key=self.api_key)
                logger.info(f"Initialized OpenAI clients with model: {model}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI clients: {e}")
                raise
    
    async def generate_completion(
        self, 
        prompt: str, 
        structured_output: bool = False,
        max_retries: int = 2
    ) -> str:
        """
        Generate a completion from the LLM.
        
        Args:
            prompt: Prompt to complete
            structured_output: Whether to request JSON output
            max_retries: Maximum number of retries on error
            
        Returns:
            Generated completion text
        """
        # If client has a generate_completion_async method, use it
        if hasattr(self.client, "generate_completion_async"):
            return await self.client.generate_completion_async(prompt)
        
        # Otherwise use our own implementation
        logger.info(f"Generating completion with model: {self.model}, length: {len(prompt)} chars")
        
        retries = 0
        while True:
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    response_format={"type": "json_object"} if structured_output else None
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
                
                # Exponential backoff
                wait_time = 2 ** retries
                logger.warning(f"Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
    
    async def generate_structured_output(
        self, 
        prompt: str, 
        output_schema: Dict[str, Any],
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Generate completion with structured JSON output.
        
        Args:
            prompt: Prompt for generation
            output_schema: Expected JSON output schema
            max_retries: Maximum retries
            
        Returns:
            Parsed JSON result
        """
        # Add schema information to the prompt
        enhanced_prompt = f"{prompt}\n\nPlease return your response as a JSON object with this structure:\n{json.dumps(output_schema, indent=2)}"
        
        # Generate completion with structured output flag
        result = await self.generate_completion(enhanced_prompt, structured_output=True, max_retries=max_retries)
        
        # Parse the result as JSON
        try:
            return json.loads(result)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            # Try to extract JSON from the response
            try:
                # Look for JSON-like content
                if '{' in result and '}' in result:
                    json_start = result.find('{')
                    json_end = result.rfind('}') + 1
                    json_content = result[json_start:json_end]
                    return json.loads(json_content)
            except:
                # Return the raw text if parsing fails
                return {"text": result, "error": "Failed to parse as JSON"}
            
            # Return the raw text if parsing fails
            return {"text": result, "error": "Failed to parse as JSON"}
    
    def generate_completion_sync(self, prompt: str, structured_output: bool = False) -> str:
        """
        Synchronous version of generate_completion.
        
        Args:
            prompt: Prompt to complete
            structured_output: Whether to request JSON output
            
        Returns:
            Generated completion text
        """
        # Create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async method in the loop
            return loop.run_until_complete(
                self.generate_completion(prompt, structured_output)
            )
        finally:
            # Close the loop
            loop.close()

    async def generate_completion_async(self, prompt, max_retries=2):
        """
        Alias for generate_completion to maintain backwards compatibility.
        
        Args:
            prompt: Prompt to complete
            max_retries: Maximum number of retries on error
            
        Returns:
            Generated completion text
        """
        return await self.generate_completion(prompt, structured_output=False, max_retries=max_retries)