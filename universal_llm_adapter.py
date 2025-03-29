"""
Universal LLM Adapter - Enhanced implementation with robust JSON parsing and error handling.
Provides a consistent interface for all components to interact with language models.
"""

import os
import logging
import json
import asyncio
import re
import time
import traceback
from typing import Optional, Dict, Any, Union, List, Callable

logger = logging.getLogger(__name__)

class LLMAdapter:
    """
    Enhanced LLM adapter with robust JSON parsing and error handling.
    Provides a consistent interface for all components to interact with language models.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: str = "gpt-3.5-turbo", 
        temperature: float = 0.2,
        llm_client = None,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        """
        Initialize the LLM adapter.
        
        Args:
            api_key: OpenAI API key (can be None if llm_client is provided)
            model: Model name for LLM
            temperature: Temperature for LLM
            llm_client: Existing LLM client (if None, one will be created)
            max_retries: Maximum number of retries for API calls
            retry_delay: Base delay between retries (in seconds)
        """
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
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
                # Delayed import to avoid requiring openai if custom client is provided
                from openai import OpenAI, AsyncOpenAI
                
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
        max_retries: Optional[int] = None
    ) -> str:
        """
        Generate a completion from the LLM with robust error handling.
        
        Args:
            prompt: Prompt to complete
            structured_output: Whether to request JSON output
            max_retries: Maximum number of retries on error (overrides instance default)
            
        Returns:
            Generated completion text
        """
        # If client has a generate_completion_async method, use it
        if hasattr(self.client, "generate_completion_async"):
            return await self.client.generate_completion_async(prompt)
        
        # Use instance default if not specified
        retries = max_retries if max_retries is not None else self.max_retries
        
        # Otherwise use our own implementation
        logger.info(f"Generating completion with model: {self.model}, length: {len(prompt)} chars")
        
        current_retry = 0
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
                current_retry += 1
                
                # Log error details
                logger.warning(f"API error ({current_retry}/{retries}): {e}")
                
                # Raise if max retries exceeded
                if current_retry >= retries:
                    logger.error(f"Max retries exceeded: {e}")
                    raise
                
                # Exponential backoff with jitter
                wait_time = self.retry_delay * (2 ** (current_retry - 1)) * (0.8 + 0.4 * (time.time() % 1))
                logger.warning(f"Retrying in {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)
    
    async def generate_structured_output(
        self, 
        prompt: str, 
        output_schema: Dict[str, Any],
        max_retries: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate completion with structured JSON output and robust parsing.
        
        Args:
            prompt: Prompt for generation
            output_schema: Expected JSON output schema
            max_retries: Maximum retries (overrides instance default)
            
        Returns:
            Parsed JSON result
        """
        # Add schema information to the prompt
        schema_description = json.dumps(output_schema, indent=2)
        enhanced_prompt = f"{prompt}\n\nPlease return your response as a JSON object with this structure:\n{schema_description}"
        
        # Generate completion with structured output flag
        result = await self.generate_completion(
            enhanced_prompt, 
            structured_output=True, 
            max_retries=max_retries
        )
        
        # Parse the result with robust error handling
        return self.parse_json(result, output_schema)
    
    def parse_json(self, text: str, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parse JSON from text with multiple fallback strategies.
        
        Args:
            text: Text to parse as JSON
            schema: Optional schema for validation and fallback
            
        Returns:
            Parsed JSON object
        """
        if not text:
            logger.warning("Empty text provided for JSON parsing")
            return {} if schema is None else self._create_schema_skeleton(schema)
        
        # Log basic info without full text to avoid cluttering logs
        logger.info(f"Parsing JSON response of length {len(text)}")
        
        # Try direct parsing first (most common case)
        try:
            parsed = json.loads(text)
            logger.info("Successfully parsed JSON directly")
            return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error at position {e.pos}: {e}")
        
        # Try to extract JSON from markdown code blocks
        try:
            json_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if json_block_match:
                extracted_json = json_block_match.group(1).strip()
                parsed = json.loads(extracted_json)
                logger.info("Successfully parsed JSON from code block")
                return parsed
        except Exception as e:
            logger.warning(f"Code block extraction failed: {e}")
        
        # Try to extract JSON object with regex
        try:
            json_object_match = re.search(r"({[\s\S]*})", text)
            if json_object_match:
                extracted_json = json_object_match.group(1).strip()
                parsed = json.loads(extracted_json)
                logger.info("Successfully parsed JSON from regex extraction")
                return parsed
        except Exception as e:
            logger.warning(f"JSON object extraction failed: {e}")
        
        # Try to fix common JSON errors
        try:
            # Fix unquoted keys with regex
            fixed_text = re.sub(r'(\s*)([a-zA-Z0-9_]+)(\s*):(\s*)', r'\1"\2"\3:\4', text)
            # Fix trailing commas
            fixed_text = re.sub(r',(\s*[}\]])', r'\1', fixed_text)
            # Try parsing again
            parsed = json.loads(fixed_text)
            logger.info("Successfully parsed JSON after fixing common errors")
            return parsed
        except Exception as e:
            logger.warning(f"JSON fixing failed: {e}")
        
        # If all parsing fails, return schema skeleton or empty dict
        if schema:
            logger.warning(f"All JSON parsing methods failed, returning schema skeleton")
            return self._create_schema_skeleton(schema)
        
        logger.warning(f"All JSON parsing methods failed, returning empty dict")
        return {}
    
    def _create_schema_skeleton(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a minimal valid structure based on schema.
        
        Args:
            schema: JSON schema
            
        Returns:
            Skeleton object matching schema
        """
        result = {}
        
        for key, value_info in schema.items():
            # Handle different schema formats
            if isinstance(value_info, dict):
                # Nested schema
                result[key] = self._create_schema_skeleton(value_info)
            elif isinstance(value_info, str):
                # String type description
                type_str = value_info.lower()
                if "array" in type_str or "list" in type_str:
                    result[key] = []
                elif "number" in type_str or "integer" in type_str:
                    result[key] = 0
                elif "boolean" in type_str:
                    result[key] = False
                else:
                    result[key] = ""
            else:
                # Default to empty string
                result[key] = ""
                
        return result
    
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

    async def generate_completion_async(self, prompt: str, max_retries: Optional[int] = None) -> str:
        """
        Alias for generate_completion to maintain backwards compatibility.
        
        Args:
            prompt: Prompt to complete
            max_retries: Maximum number of retries on error
            
        Returns:
            Generated completion text
        """
        return await self.generate_completion(prompt, structured_output=False, max_retries=max_retries)
    
    async def chat_with_context(
        self, 
        user_message: str, 
        chat_history: List[Dict[str, str]],
        document_text: Optional[str] = None,
        summary_text: Optional[str] = None,
        document_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a response with conversation history and document context.
        
        Args:
            user_message: Current user message
            chat_history: Previous messages in the conversation
            document_text: Optional document text for context (may be truncated)
            summary_text: Optional summary of the document (preferred over full text)
            document_info: Optional document metadata
            
        Returns:
            Generated assistant response
        """
        # Prepare conversation history
        messages = []
        
        # Add system prompt with document context
        system_message = self._build_system_prompt(document_text, summary_text, document_info)
        messages.append({"role": "system", "content": system_message})
        
        # Add conversation history (limit to reasonable number of messages)
        for msg in chat_history[-10:]:  # Only include last 10 messages
            if msg and isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append(msg)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        try:
            # Generate response
            logger.info(f"Generating chat response with {len(messages)} messages")
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            
            result = response.choices[0].message.content
            logger.info(f"Chat response generated, length: {len(result)} chars")
            return result
            
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            return f"I apologize, but I encountered an error processing your request: {str(e)}"
    
    def _build_system_prompt(
        self, 
        document_text: Optional[str] = None,
        summary_text: Optional[str] = None,
        document_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build a system prompt with document context.
        
        Args:
            document_text: Optional document text
            summary_text: Optional document summary
            document_info: Optional document metadata
            
        Returns:
            System prompt string
        """
        # Base system prompt
        system_prompt = (
            "You are a helpful assistant that answers questions about documents. "
            "Base your responses on the information provided in the context."
        )
        
        # Add document summary if available (preferred over full text)
        if summary_text:
            # Truncate if too long
            if len(summary_text) > 4000:
                logger.info(f"Truncating summary text from {len(summary_text)} chars to 4000")
                summary_text = summary_text[:4000] + "..."
                
            system_prompt += f"\n\nDOCUMENT SUMMARY:\n{summary_text}"
        
        # Add document info if available
        elif document_info:
            system_prompt += "\n\nDOCUMENT INFORMATION:\n"
            
            # Add preview analysis if available
            if "preview_analysis" in document_info:
                preview = document_info["preview_analysis"]
                
                if "summary" in preview:
                    system_prompt += f"Summary: {preview['summary']}\n"
                
                if "key_topics" in preview and preview["key_topics"]:
                    topics = ", ".join(preview["key_topics"])
                    system_prompt += f"Key Topics: {topics}\n"
                
                if "domain_categories" in preview and preview["domain_categories"]:
                    categories = ", ".join(preview["domain_categories"])
                    system_prompt += f"Categories: {categories}\n"
            
            # Add basic stats if available
            if "basic_stats" in document_info:
                stats = document_info["basic_stats"]
                system_prompt += f"Word Count: {stats.get('word_count', 'Unknown')}\n"
                system_prompt += f"Paragraph Count: {stats.get('paragraph_count', 'Unknown')}\n"
        
        # Add document text as a last resort (truncated)
        elif document_text:
            # Truncate if too long
            if len(document_text) > 3000:
                logger.info(f"Truncating document text from {len(document_text)} chars to 3000")
                document_text = document_text[:3000] + "...\n[Document truncated due to length]"
                
            system_prompt += f"\n\nDOCUMENT TEXT:\n{document_text}"
        
        # Add guidance for handling unknown information
        system_prompt += (
            "\n\nIf you don't know the answer based on the provided information, "
            "acknowledge that and avoid making up facts. You can suggest what additional "
            "information would be needed to answer the question."
        )
        
        return system_prompt
    
    async def process_with_retry(
        self, 
        func: Callable, 
        *args, 
        max_retries: Optional[int] = None, 
        **kwargs
    ) -> Any:
        """
        Execute a function with retry logic for robust processing.
        
        Args:
            func: Function to execute
            *args: Arguments for the function
            max_retries: Maximum retries (overrides instance default)
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result
        """
        # Use instance default if not specified
        retries = max_retries if max_retries is not None else self.max_retries
        
        current_retry = 0
        last_error = None
        
        while True:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                current_retry += 1
                last_error = e
                
                # Log error details
                logger.warning(f"Error in function {func.__name__} (retry {current_retry}/{retries}): {e}")
                
                # Raise if max retries exceeded
                if current_retry >= retries:
                    logger.error(f"Max retries exceeded for {func.__name__}: {e}")
                    raise
                
                # Exponential backoff with jitter
                wait_time = self.retry_delay * (2 ** (current_retry - 1)) * (0.8 + 0.4 * (time.time() % 1))
                logger.warning(f"Retrying {func.__name__} in {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)