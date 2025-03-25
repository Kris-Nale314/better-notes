"""
Base agent module for Better Notes with cleaner architecture.
Provides the foundation for all specialized agents in the system.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import asyncio
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class BaseAgent:
    """
    Base class for all specialized agents in Better Notes.
    Simplified with clearer configuration access and improved error handling.
    """
    
    def __init__(
        self,
        llm_client,
        agent_type: str,
        crew_type: str,
        config: Optional[Dict[str, Any]] = None,
        config_manager = None,
        verbose: bool = True,
        max_chunk_size: int = 1500,
        max_rpm: int = 10
    ):
        """
        Initialize a base agent with simpler configuration access.
        
        Args:
            llm_client: LLM client for API calls
            agent_type: Type of agent (planner, extractor, etc.)
            crew_type: Type of crew (issues, actions, etc.)
            config: Optional pre-loaded configuration
            config_manager: Optional config manager
            verbose: Whether to enable verbose logging
            max_chunk_size: Maximum size of text chunks to process
            max_rpm: Maximum requests per minute for API rate limiting
        """
        # Store core properties
        self.llm_client = llm_client
        self.agent_type = agent_type
        self.crew_type = crew_type
        self.verbose = verbose
        self.max_chunk_size = max_chunk_size
        self.max_rpm = max_rpm
        
        # Load or get config manager
        if config_manager:
            self.config_manager = config_manager
        else:
            # Import here to avoid circular imports
            from config_manager import ConfigManager
            self.config_manager = ConfigManager()
        
        # Load config if not provided
        self.config = config if config else self.config_manager.get_config(crew_type)
        
        # Execution tracking
        self.stats = {
            "runs": 0,
            "total_time": 0,
            "avg_time": 0,
            "last_run": None
        }
        
        if self.verbose:
            logger.info(f"Initialized {self.agent_type} agent for {self.crew_type} crew")
    
    async def process(self, context):
        """
        Process input using this agent with the given context.
        To be implemented by subclasses.
        
        Args:
            context: ProcessingContext object
            
        Returns:
            Processing result
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the process method")
    
    def get_role_description(self) -> str:
        """
        Get the description of this agent's role from config.
        
        Returns:
            Role description string
        """
        return self.config.get("workflow", {}).get("agent_roles", {}).get(
            self.agent_type, {}).get("description", f"{self.agent_type.capitalize()} Agent")
    
    def get_primary_task(self) -> str:
        """
        Get the primary task for this agent from config.
        
        Returns:
            Primary task string
        """
        return self.config.get("workflow", {}).get("agent_roles", {}).get(
            self.agent_type, {}).get("primary_task", f"Process {self.crew_type} data")
    
    def get_output_schema(self) -> Dict[str, Any]:
        """
        Get the output schema for this agent from config.
        
        Returns:
            Output schema dictionary
        """
        return self.config.get("workflow", {}).get("agent_roles", {}).get(
            self.agent_type, {}).get("output_schema", {})
    
    def get_instructions(self, context) -> str:
        """
        Get instructions for this agent, prioritizing context instructions.
        
        Args:
            context: ProcessingContext object
            
        Returns:
            Instructions string
        """
        # First try to get from context
        if hasattr(context, 'agent_instructions') and self.agent_type in context.agent_instructions:
            instructions = context.agent_instructions[self.agent_type].get('instructions')
            if instructions:
                return instructions
        
        # Fall back to primary task from config
        return self.get_primary_task()
    
    def get_emphasis(self, context) -> str:
        """
        Get emphasis for this agent from context.
        
        Args:
            context: ProcessingContext object
            
        Returns:
            Emphasis string or empty string if not found
        """
        if hasattr(context, 'agent_instructions') and self.agent_type in context.agent_instructions:
            return context.agent_instructions[self.agent_type].get('emphasis', '')
        return ''
    
    def build_prompt(self, context) -> str:
        """
        Build a prompt for this agent using the context.
        
        Args:
            context: ProcessingContext object
            
        Returns:
            Formatted prompt string
        """
        # Get instructions
        instructions = self.get_instructions(context)
        
        # Get role description
        role_description = self.get_role_description()
        
        # Get emphasis if available
        emphasis = self.get_emphasis(context)
        
        # Get output schema info
        output_schema = self.get_output_schema()
        schema_info = ""
        if output_schema:
            schema_info = f"\n\nOUTPUT SCHEMA:\n{json.dumps(output_schema, indent=2)}"
        
        # Start with role and instructions
        prompt = f"You are a {role_description}.\n\nTASK:\n{instructions}\n\n"
        
        # Add emphasis if available
        if emphasis:
            prompt += f"EMPHASIS:\n{emphasis}\n\n"
        
        # Add context information
        if hasattr(context, 'document_info') and context.document_info:
            # Include only essential document info to save tokens
            essential_info = self._extract_essential_info(context.document_info)
            prompt += f"\nDOCUMENT INFO:\n{json.dumps(essential_info, indent=2)}\n\n"
        
        if hasattr(context, 'options') and context.options:
            # Include relevant user preferences
            user_prefs = self._extract_user_preferences(context.options)
            prompt += f"\nUSER PREFERENCES:\n{json.dumps(user_prefs, indent=2)}\n\n"
        
        # Add stage-specific content (to be overridden by subclasses)
        stage_content = self._get_stage_specific_content(context)
        if stage_content:
            prompt += f"{stage_content}\n\n"
        
        # Add output schema if available
        if schema_info:
            prompt += schema_info
            prompt += "\n\nYour response MUST follow the specified output schema."
        
        # Ensure prompt doesn't exceed max size
        return self.truncate_text(prompt, 8000)
    
    def _extract_essential_info(self, document_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract only essential information from document_info to save tokens.
        
        Args:
            document_info: Full document information
            
        Returns:
            Dictionary with essential information
        """
        if not document_info:
            return {}
            
        essential = {}
        
        # Copy important fields
        if "is_meeting_transcript" in document_info:
            essential["is_meeting_transcript"] = document_info["is_meeting_transcript"]
            
        if "basic_stats" in document_info:
            essential["word_count"] = document_info["basic_stats"].get("word_count", 0)
            essential["sentence_count"] = document_info["basic_stats"].get("sentence_count", 0)
            
        # Add preview analysis if available
        if "preview_analysis" in document_info:
            preview = document_info["preview_analysis"]
            
            if "summary" in preview:
                essential["summary"] = preview["summary"]
                
            if "key_topics" in preview:
                essential["key_topics"] = preview["key_topics"][:5]  # Only include top 5
        
        return essential
    
    def _extract_user_preferences(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant user preferences from options.
        
        Args:
            options: Full options dictionary
            
        Returns:
            Dictionary with relevant preferences
        """
        if not options:
            return {}
            
        relevant = {}
        
        # Copy relevant fields
        for key in ["detail_level", "focus_areas", "user_instructions"]:
            if key in options:
                relevant[key] = options[key]
        
        return relevant
    
    def _get_stage_specific_content(self, context) -> str:
        """
        Get stage-specific content for the prompt.
        To be overridden by subclasses.
        
        Args:
            context: ProcessingContext object
            
        Returns:
            Stage-specific content string
        """
        return ""
    
    @contextmanager
    def execution_tracking(self):
        """
        Context manager for tracking execution metrics.
        Automatically handles timing and error logging.
        """
        start_time = datetime.now()
        try:
            yield
        except Exception as e:
            logger.error(f"Error in {self.agent_type} agent: {str(e)}")
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(execution_time, error=str(e))
            raise
        else:
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(execution_time)
    
    async def execute_task(self, context) -> Any:
        """
        Execute a task using this agent.
        Uses execution tracking for monitoring.
        
        Args:
            context: ProcessingContext object or dict
            
        Returns:
            Task result
        """
        with self.execution_tracking():
            # Build the prompt
            prompt = self.build_prompt(context)
            
            # Get output schema
            output_schema = self.get_output_schema()
            
            if self.verbose:
                logger.info(f"Executing {self.agent_type} agent with prompt length {len(prompt)}")
            
            # Execute based on whether we need structured output
            if output_schema and hasattr(self.llm_client, 'generate_structured_output'):
                # Use structured output if supported
                result = await self.llm_client.generate_structured_output(prompt, output_schema)
            else:
                # Use standard completion
                result = await self.llm_client.generate_completion_async(prompt)
                
                # Try to parse as JSON if we have a schema
                if output_schema and isinstance(result, str):
                    try:
                        result = self.parse_llm_json(result)
                    except:
                        if self.verbose:
                            logger.warning(f"{self.agent_type} agent response could not be parsed as JSON")
            
            return result
    
    def _update_stats(self, execution_time, error=None):
        """
        Update the agent's execution statistics.
        
        Args:
            execution_time: Execution time in seconds
            error: Optional error message
        """
        self.stats["runs"] += 1
        self.stats["last_run"] = datetime.now().isoformat()
        self.stats["total_time"] += execution_time
        self.stats["avg_time"] = self.stats["total_time"] / self.stats["runs"]
        
        # Log execution stats periodically
        if self.verbose and self.stats["runs"] % 5 == 0:
            logger.info(
                f"{self.agent_type.title()} agent stats: {self.stats['runs']} runs, "
                f"avg time: {self.stats['avg_time']:.2f}s"
            )
        
        # Log errors
        if error and self.verbose:
            logger.error(f"{self.agent_type.title()} agent execution error: {error}")
    
    def truncate_text(self, text: str, max_length: int = None) -> str:
        """
        Truncate text to a maximum length with intelligent splitting.
        
        Args:
            text: Text to truncate
            max_length: Maximum length (defaults to self.max_chunk_size)
            
        Returns:
            Truncated text
        """
        if max_length is None:
            max_length = self.max_chunk_size
            
        if not isinstance(text, str) or len(text) <= max_length:
            return text
        
        # Keep most of the beginning and some of the end
        prefix_size = int(max_length * 0.7)
        suffix_size = max_length - prefix_size - 30  # Allow for middle message
        
        # Find sentence boundaries for cleaner breaks
        prefix_end = prefix_size
        suffix_start = len(text) - suffix_size
        
        # Try to find a sentence boundary for prefix
        sentence_end = text.rfind('. ', prefix_size - 200, prefix_size + 200)
        if sentence_end > 0:
            prefix_end = sentence_end + 1
        
        # Try to find a sentence boundary for suffix
        sentence_start = text.find('. ', suffix_start - 200, suffix_start + 200)
        if sentence_start > 0:
            suffix_start = sentence_start + 2
        
        # Combine the pieces
        truncated = (
            text[:prefix_end] + 
            f"\n\n[...{len(text) - prefix_end - (len(text) - suffix_start)} characters omitted...]\n\n" + 
            text[suffix_start:]
        )
        
        return truncated
    
    def parse_llm_json(self, text, default_value=None):
        """
        Parse JSON from LLM response with fallbacks.
        
        Args:
            text: Text containing JSON
            default_value: Default value to return if parsing fails
            
        Returns:
            Parsed JSON object or default value
        """
        if not text:
            return default_value or {}
            
        # Try direct parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from text with brace matching
        try:
            # Find outermost braces
            start = text.find('{')
            if start >= 0:
                brace_count = 1
                for i in range(start + 1, len(text)):
                    if text[i] == '{':
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i + 1
                            return json.loads(text[start:end])
        except:
            pass
            
        # Try fixing common JSON issues
        try:
            # Fix unquoted keys
            fixed = re.sub(r'(\s*)(\w+)(\s*):(\s*)', r'\1"\2"\3:\4', text)
            # Fix trailing commas
            fixed = re.sub(r',(\s*[}\]])', r'\1', text)
            return json.loads(fixed)
        except:
            pass
            
        # Try extracting JSON using regex
        try:
            import re
            json_match = re.search(r'({[\s\S]*})', text)
            if json_match:
                return json.loads(json_match.group(1))
        except:
            pass
        
        # If all parsing fails, return default
        return default_value or {}