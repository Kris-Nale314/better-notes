"""
Enhanced BaseAgent for Better Notes with improved error handling and JSON parsing.
Provides the foundation for all specialized agents in the system.
"""

import logging
import json
import re
import time
import traceback
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class BaseAgent:
    """
    Enhanced base class for all specialized agents in Better Notes.
    Features improved error handling, JSON parsing, and logging.
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
        Initialize a base agent.
        
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
            self.config_manager = ConfigManager.get_instance()
        
        # Load config if not provided
        if config is not None:
            self.config = config
            logger.info(f"{self.agent_type} agent using provided config with {len(config)} keys")
        else:
            self.config = self.config_manager.get_config(crew_type)
            logger.info(f"{self.agent_type} agent loaded config for {crew_type}")
        
        # Validate important config sections
        self._validate_config()
        
        # Execution tracking
        self.stats = {
            "runs": 0,
            "total_time": 0,
            "avg_time": 0,
            "last_run": None,
            "errors": 0
        }
        
        if self.verbose:
            logger.info(f"Initialized {self.agent_type} agent for {self.crew_type} crew")
    
    def _validate_config(self):
        """Validate important config sections."""
        if not isinstance(self.config, dict):
            logger.error(f"Invalid config for {self.agent_type} agent: not a dictionary")
            self.config = {}
            return
            
        if "workflow" not in self.config:
            logger.warning(f"Missing 'workflow' in config for {self.agent_type} agent")
            self.config["workflow"] = {}
            
        if "agent_roles" not in self.config.get("workflow", {}):
            logger.warning(f"Missing 'agent_roles' in workflow for {self.agent_type} agent")
            self.config["workflow"]["agent_roles"] = {}
            
        if self.agent_type not in self.config.get("workflow", {}).get("agent_roles", {}):
            logger.warning(f"No configuration for {self.agent_type} agent in agent_roles")
            self.config["workflow"]["agent_roles"][self.agent_type] = {
                "description": f"{self.agent_type.capitalize()} Agent",
                "primary_task": f"Process {self.crew_type} data"
            }
    
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
            prompt += "\n\nYour response MUST follow the specified output schema. Return ONLY valid JSON without explanation."
        
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
            logger.error(traceback.format_exc())
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(execution_time, error=str(e))
            raise
        else:
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(execution_time)
    
    async def execute_task(self, context) -> Any:
        """
        Execute a task using this agent with improved error handling.
        
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
                try:
                    result = await self.llm_client.generate_structured_output(prompt, output_schema)
                    logger.info(f"{self.agent_type} agent generated structured output")
                    return result
                except Exception as e:
                    logger.warning(f"Error generating structured output: {e}, falling back to standard completion")
                    # Fall back to standard completion with manual parsing
                    result = await self.llm_client.generate_completion_async(prompt)
                    return self.parse_llm_json(result, output_schema)
            else:
                # Use standard completion
                result = await self.llm_client.generate_completion_async(prompt)
                
                # Try to parse as JSON if we have a schema
                if output_schema and isinstance(result, str):
                    return self.parse_llm_json(result, output_schema)
                
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
        
        # Track errors
        if error:
            self.stats["errors"] = self.stats.get("errors", 0) + 1
        
        # Log execution stats periodically
        if self.verbose and self.stats["runs"] % 5 == 0:
            logger.info(
                f"{self.agent_type.title()} agent stats: {self.stats['runs']} runs, "
                f"avg time: {self.stats['avg_time']:.2f}s, errors: {self.stats.get('errors', 0)}"
            )
    
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
    
    def parse_llm_json(self, text: str, default_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parse JSON from LLM response with improved error handling.
        
        Args:
            text: Text containing JSON
            default_schema: Default schema for fallback
            
        Returns:
            Parsed JSON object
        """
        if not text:
            logger.warning(f"{self.agent_type} agent received empty response")
            return {} if default_schema is None else self._create_schema_skeleton(default_schema)
        
        # Log basic info without full text to avoid cluttering logs
        logger.info(f"{self.agent_type} agent parsing JSON response of length {len(text)}")
        
        # Try direct JSON parsing first (most common case)
        try:
            parsed = json.loads(text)
            logger.info(f"{self.agent_type} agent successfully parsed JSON directly")
            return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error at position {e.pos}: {e}")
        
        # Try to extract JSON from markdown code blocks
        try:
            json_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if json_block_match:
                extracted_json = json_block_match.group(1).strip()
                parsed = json.loads(extracted_json)
                logger.info(f"{self.agent_type} agent parsed JSON from code block")
                return parsed
        except Exception as e:
            logger.warning(f"Code block extraction failed: {e}")
        
        # Try to extract JSON object with regex
        try:
            json_object_match = re.search(r"({[\s\S]*})", text)
            if json_object_match:
                extracted_json = json_object_match.group(1).strip()
                parsed = json.loads(extracted_json)
                logger.info(f"{self.agent_type} agent parsed JSON from regex extraction")
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
            logger.info(f"{self.agent_type} agent parsed JSON after fixing common errors")
            return parsed
        except Exception as e:
            logger.warning(f"JSON fixing failed: {e}")
        
        # If all parsing fails, return schema skeleton or empty dict
        if default_schema:
            logger.warning(f"All JSON parsing methods failed, returning schema skeleton")
            return self._create_schema_skeleton(default_schema)
        
        logger.warning(f"All JSON parsing methods failed, returning empty dict")
        return {}
    
    def _create_schema_skeleton(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a minimal valid structure based on schema.
        
        Args:
            schema: Schema dictionary
            
        Returns:
            Skeleton dictionary matching schema
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