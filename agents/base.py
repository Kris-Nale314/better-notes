# agents/base.py
"""
Enhanced base agent module with ProcessingContext support.
Provides the foundation for all specialized agents in the system.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import asyncio


logger = logging.getLogger(__name__)

class BaseAgent:
    """
    Enhanced base class for all specialized agents in Better Notes.
    Designed to work with ProcessingContext for improved state management.
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
        Initialize a base agent with enhanced configuration support.
        
        Args:
            llm_client: LLM client for API calls
            agent_type: Type of agent (extraction, aggregation, evaluation, etc.)
            crew_type: Type of crew (issues, actions, etc.)
            config: Optional pre-loaded configuration
            config_manager: Optional config manager
            verbose: Whether to enable verbose logging
            max_chunk_size: Maximum size of text chunks to process
            max_rpm: Maximum requests per minute for API rate limiting
        """
        # Store adapter and configuration
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
        self.execution_stats = {
            "runs": 0,
            "total_execution_time": 0,
            "average_execution_time": 0,
            "last_run_time": None
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
        raise NotImplementedError("Each agent must implement the process method")
    
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
        
        # Fall back to config
        agent_config = self._get_agent_config()
        if "instructions" in agent_config:
            return agent_config["instructions"]
        
        # Default instructions
        return f"Process the {self.crew_type} as a {self.agent_type} agent."
    
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
    
    def _get_agent_config(self) -> Dict[str, Any]:
        """
        Get configuration for this agent type.
        
        Returns:
            Agent configuration dictionary
        """
        # Check the agents section first
        if "agents" in self.config and self.agent_type in self.config["agents"]:
            return self.config["agents"][self.agent_type]
        
        # Fall back to legacy structure
        if self.agent_type in self.config:
            return self.config[self.agent_type]
        
        # Return empty dict if no config found
        return {}
    
    def get_output_format(self) -> Dict[str, Any]:
        """
        Get the expected output format for this agent.
        
        Returns:
            Output format dictionary
        """
        agent_config = self._get_agent_config()
        return agent_config.get("output_format", {})
    
    def build_prompt(self, context) -> str:
        """
        Build a prompt for this agent using the context.
        
        Args:
            context: ProcessingContext object
            
        Returns:
            Formatted prompt string
        """
        # Get base instructions
        instructions = self.get_instructions(context)
        
        # Get emphasis if available
        emphasis = self.get_emphasis(context)
        
        # Get output format info
        output_format = self.get_output_format()
        format_info = ""
        if output_format:
            format_info = f"\n\nOUTPUT FORMAT:\n{json.dumps(output_format, indent=2)}"
        
        # Start with the instructions
        prompt = f"{instructions}\n\n"
        
        # Add emphasis if available
        if emphasis:
            prompt += f"SPECIAL EMPHASIS:\n{emphasis}\n\n"
        
        # Add context information
        if hasattr(context, 'document_info') and context.document_info:
            prompt += f"\nDOCUMENT INFO:\n{json.dumps(context.document_info, indent=2, default=str)[:1000]}\n"
        
        if hasattr(context, 'options') and context.options:
            prompt += f"\nUSER PREFERENCES:\n{json.dumps(context.options, indent=2, default=str)[:500]}\n"
        
        # Add stage-specific content (to be overridden by subclasses)
        stage_content = self._get_stage_specific_content(context)
        if stage_content:
            prompt += f"\n{stage_content}\n"
        
        # Add output format if available
        if format_info:
            prompt += format_info
        
        # Ensure prompt doesn't exceed max size
        return self.truncate_text(prompt, 6000)
    
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
    
    async def execute_task(self, context) -> Any:
        """
        Execute a task using this agent.
        
        Args:
            context: ProcessingContext object
            
        Returns:
            Task result
        """
        # Start timing
        start_time = datetime.now()
        
        try:
            # Build the prompt from context
            prompt = self.build_prompt(context)
            prompt_length = len(prompt)
            
            # Log execution
            if self.verbose:
                logger.info(f"Executing {self.agent_type} agent with prompt length {prompt_length}")
            
            # Check if structured output is needed
            output_format = self.get_output_format()
            structured_output = bool(output_format)
            
            # Execute the task
            if structured_output and hasattr(self.llm_client, 'generate_structured_output'):
                # Use structured output mode
                result = await self.llm_client.generate_structured_output(prompt, output_format)
            else:
                # Use standard completion
                result = await self.llm_client.generate_completion_async(prompt)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update stats
            self._update_execution_stats(execution_time, prompt_length, len(str(result)))
            
            # Return the result
            return result
            
        except Exception as e:
            # Handle errors
            logger.error(f"Error executing task with {self.agent_type} agent: {str(e)}")
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update stats with error
            self._update_execution_stats(
                execution_time, 
                len(prompt) if 'prompt' in locals() else 0, 
                0, 
                error=str(e)
            )
            
            # Return error information
            return {
                "error": str(e),
                "agent_type": self.agent_type
            }
    
    def _update_execution_stats(self, execution_time, prompt_length, result_length, error = None):
        """Update the agent's execution statistics."""
        self.execution_stats["runs"] += 1
        self.execution_stats["last_run_time"] = datetime.now().isoformat()
        self.execution_stats["total_execution_time"] += execution_time
        self.execution_stats["average_execution_time"] = (
            self.execution_stats["total_execution_time"] / self.execution_stats["runs"]
        )
        
        # Log execution stats periodically
        if self.verbose and self.execution_stats["runs"] % 5 == 0:
            logger.info(
                f"{self.agent_type.title()} agent stats: {self.execution_stats['runs']} runs, "
                f"avg time: {self.execution_stats['average_execution_time']:.2f}s"
            )
        
        # Log errors
        if error and self.verbose:
            logger.error(f"{self.agent_type.title()} agent execution error: {error}")
    
    def truncate_text(self, text: str, max_length: int = None) -> str:
        """
        Truncate text to a maximum length.
        
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
        
        # Enhanced smart truncation with section awareness
        prefix_size = int(max_length * 0.7)  # Prioritize the beginning
        suffix_size = int(max_length * 0.25)  # Keep some of the end
        middle_note_size = max_length - prefix_size - suffix_size
        
        # Look for reasonable break points near the boundaries
        prefix_end = prefix_size
        reasonable_break_chars = ["\n\n", "\n", ". ", "! ", "? ", ": ", ";\n", ";\t", ";\r"]
        
        # Try to find a clean break point for the prefix
        for break_char in reasonable_break_chars:
            pos = text.rfind(break_char, prefix_size - 100, prefix_size + 100)
            if pos > 0:
                prefix_end = pos + len(break_char)
                break
        
        # Try to find a clean start point for the suffix
        suffix_start = len(text) - suffix_size
        for break_char in reasonable_break_chars:
            pos = text.find(break_char, suffix_start - 100, suffix_start + 100)
            if pos > 0:
                suffix_start = pos
                break
        
        # Create a meaningful middle note
        middle_note = f"\n\n[...{len(text) - prefix_end - (len(text) - suffix_start)} characters truncated...]\n\n"
        
        # Combine the pieces
        return text[:prefix_end] + middle_note + text[suffix_start:]