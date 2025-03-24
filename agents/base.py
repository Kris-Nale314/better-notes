"""
Enhanced base agent module with improved metadata handling and configuration support.
Provides the foundation for all specialized agents in the system.
"""

from typing import Dict, Any, Optional, List, Union
import os
import json
import logging
from datetime import datetime

# Import our universal adapter and config manager
from universal_llm_adapter import UniversalLLMAdapter
from config_manager import ConfigManager

logger = logging.getLogger(__name__)

class BaseAgent:
    """
    Enhanced base class for all specialized agents in Better Notes.
    Provides common functionality and configuration loading with support
    for the enhanced metadata structure and instruction flow.
    """
    
    def __init__(
        self,
        llm_client,
        agent_type: str,
        crew_type: str,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        max_chunk_size: int = 1500,
        max_rpm: int = 10,
        custom_instructions: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a base agent with enhanced configuration support.
        
        Args:
            llm_client: LLM client from orchestrator
            agent_type: Type of agent (extraction, aggregation, evaluation, formatting, reviewer)
            crew_type: Type of crew (issues, actions, opportunities)
            config: Optional pre-loaded configuration
            verbose: Whether to enable verbose mode
            max_chunk_size: Maximum size of text chunks to process
            max_rpm: Maximum requests per minute for API rate limiting
            custom_instructions: Optional custom instructions from Instructor agent
        """
        # Ensure we have a UniversalLLMAdapter
        if not isinstance(llm_client, UniversalLLMAdapter):
            self.llm_client = UniversalLLMAdapter(llm_client=llm_client)
        else:
            self.llm_client = llm_client
            
        self.agent_type = agent_type
        self.crew_type = crew_type
        self.verbose = verbose
        self.max_chunk_size = max_chunk_size
        self.max_rpm = max_rpm
        self.custom_instructions = custom_instructions or {}
        
        # Create config manager for loading configs
        self.config_manager = ConfigManager()
        
        # Execution tracking
        self.execution_stats = {
            "runs": 0,
            "last_run_time": None,
            "total_execution_time": 0,
            "average_execution_time": 0,
            "last_execution_metadata": None
        }
        
        # Load or use provided config
        self.config = config if config else self.load_config(crew_type)
        
        # Get agent-specific configuration from the new structure
        agent_config = self._get_agent_config()
        
        # Create the CrewAI agent with optimized settings
        try:
            # Try to import CrewAI
            from crewai import Agent
            
            self.agent = Agent(
                role=agent_config.get("role", f"{crew_type.title()} {agent_type.title()} Agent"),
                goal=agent_config.get("goal", f"Process {crew_type} through {agent_type}"),
                backstory=self._get_backstory(agent_config),
                verbose=verbose,
                llm=llm_client,
                # Add specific settings to help with large contexts
                max_iterations=2,
                max_rpm=max_rpm, 
                llm_config={
                    "max_tokens": 800,  # Increased from 600 for more complex responses
                    "temperature": llm_client.temperature
                }
            )
        except ImportError:
            logger.warning("CrewAI not available, using direct LLM calls")
            self.agent = None
    
    def _get_agent_config(self) -> Dict[str, Any]:
        """
        Get configuration for this agent from the new structure.
        
        Returns:
            Agent configuration dictionary
        """
        # Try to get from new structure first
        if "agents" in self.config and self.agent_type in self.config["agents"]:
            return self.config["agents"][self.agent_type]
        
        # Fall back to old structure
        agent_config = self.config.get("agent_config", {}).get(self.agent_type, {})
        if agent_config:
            return agent_config
        
        # Use defaults if neither found
        return self._get_default_agent_config()
    
    def _get_backstory(self, agent_config: Dict[str, Any]) -> str:
        """
        Get an optimized backstory for the agent.
        
        Args:
            agent_config: Agent configuration dictionary
            
        Returns:
            Optimized backstory string
        """
        # Check for backstory in config
        backstory = agent_config.get("backstory", "")
        
        # If no backstory found, construct from role and goal
        if not backstory:
            role = agent_config.get("role", f"{self.crew_type.title()} {self.agent_type.title()} Agent")
            goal = agent_config.get("goal", f"Process {self.crew_type} through {self.agent_type}")
            backstory = f"I am a {role} specializing in {self.crew_type}. My goal is to {goal}."
        
        # Compact backstory to avoid token wastage
        return backstory[:200] if len(backstory) > 200 else backstory
    
    def load_config(self, crew_type: str) -> Dict[str, Any]:
        """
        Load a configuration file for the specified crew type.
        
        Args:
            crew_type: Type of crew (issues, actions, opportunities)
            
        Returns:
            Configuration dictionary
        """
        return self.config_manager.get_config(crew_type)
    
    def _get_default_agent_config(self) -> Dict[str, Any]:
        """
        Get default configuration for an agent if not specified in config file.
        
        Returns:
            Default configuration dictionary
        """
        # Default roles and goals
        defaults = {
            "extraction": {
                "role": f"{self.crew_type.title()} Extractor",
                "goal": f"Identify all potential {self.crew_type} in document chunks and add initial metadata",
                "instructions": f"Analyze the document to identify {self.crew_type} with detailed metadata."
            },
            "aggregation": {
                "role": f"{self.crew_type.title()} Aggregator",
                "goal": f"Combine and deduplicate {self.crew_type} from multiple extractions while enhancing metadata",
                "instructions": f"Combine similar {self.crew_type} while preserving important distinctions and track mention frequency."
            },
            "evaluation": {
                "role": f"{self.crew_type.title()} Evaluator",
                "goal": f"Assess importance and impact of identified {self.crew_type} and add impact assessment metadata",
                "instructions": f"Evaluate each {self.crew_type.rstrip('s')} for importance, priority, and potential impact."
            },
            "formatting": {
                "role": "Report Formatter",
                "goal": "Create clear, structured reports from analysis results with enhanced navigation",
                "instructions": f"Format the {self.crew_type} into a clear, well-organized report with visual priorities."
            },
            "reviewer": {
                "role": "Analysis Reviewer",
                "goal": "Ensure analysis meets quality standards and user expectations",
                "instructions": "Review the report for quality, consistency, and alignment with user needs."
            },
            "instructor": {
                "role": "Instruction Architect",
                "goal": "Create tailored instructions for each agent based on document and user preferences",
                "instructions": "Create specialized instructions for each agent in the workflow."
            }
        }
        
        return defaults.get(self.agent_type, {})
    
    def get_instructions(self) -> str:
        """
        Get instructions for this agent, prioritizing custom instructions from Planner.
        
        Returns:
            Instructions string
        """
        # First check for custom instructions from Instructor
        if self.custom_instructions and "instructions" in self.custom_instructions:
            return self.custom_instructions["instructions"]
        
        # Then check for instructions in new config structure
        agent_config = self._get_agent_config()
        if "instructions" in agent_config:
            return agent_config["instructions"]
        
        # Fall back to prompt template from old structure
        return self.config.get(self.agent_type, {}).get("prompt_template", "")
    
    def get_output_format(self) -> Dict[str, Any]:
        """
        Get the expected output format for this agent.
        
        Returns:
            Output format dictionary
        """
        # Get from new config structure
        agent_config = self._get_agent_config()
        if "output_format" in agent_config:
            return agent_config["output_format"]
        
        # Fall back to output schema from old structure
        return self.config.get(self.agent_type, {}).get("output_schema", {})
    
    def get_template(self) -> str:
        """
        Get the template for formatting agents.
        
        Returns:
            Template string
        """
        if self.agent_type == "formatting":
            # First check new structure
            agent_config = self._get_agent_config()
            if "html_template" in agent_config:
                return agent_config["html_template"]
            
            # Fall back to old structure
            return self.config.get("formatting", {}).get("format_template", "")
        return ""
    
    def build_prompt(self, context: Dict[str, Any] = None) -> str:
        """
        Build a prompt using instructions and context.
        
        Args:
            context: Dictionary of context variables to fill the prompt
            
        Returns:
            Formatted prompt string
        """
        # Get base instructions
        instructions = self.get_instructions()
        context = context or {}
        
        # Check for special emphasis from Instructor
        emphasis = ""
        if self.custom_instructions and "emphasis" in self.custom_instructions:
            emphasis = self.custom_instructions["emphasis"]
        
        # Get output format info
        output_format = self.get_output_format()
        format_info = ""
        if output_format:
            format_info = f"\n\nOUTPUT FORMAT:\n{json.dumps(output_format, indent=2)}"
        
        # Create a new prompt with instructions, emphasis, and format info
        prompt = f"{instructions}\n\n"
        
        if emphasis:
            prompt += f"SPECIAL EMPHASIS:\n{emphasis}\n\n"
        
        # Add formatted context
        for key, value in context.items():
            prompt += f"\n{key.upper()}:\n"
            if isinstance(value, str):
                value = self.truncate_text(value, 2000)
                prompt += value + "\n"
            else:
                # Format complex objects
                try:
                    # Try to serialize to JSON with indent
                    json_str = json.dumps(value, indent=2, default=str)[:1500]
                    prompt += json_str + "\n"
                except:
                    # Fall back to string representation
                    prompt += str(value)[:1000] + "\n"
        
        # Add output format if available
        if format_info:
            prompt += format_info
        
        # Ensure final prompt isn't too large
        return self.truncate_text(prompt, 6000)  # Increased from 4000 to handle larger contexts
    
    def execute_task(self, description: str = None, context: Dict[str, Any] = None) -> Any:
        """
        Execute a task using this agent with enhanced metadata tracking.
        
        Args:
            description: Optional task description (overrides built prompt)
            context: Context for building the prompt if description not provided
            
        Returns:
            Task result
        """
        # Start timing execution
        start_time = datetime.now()
        execution_id = f"{self.agent_type}-{start_time.strftime('%Y%m%d-%H%M%S')}"
        
        # Build the description if not provided
        if not description and context:
            description = self.build_prompt(context)
        elif not description:
            description = self.build_prompt()
        
        # Ensure description isn't too large
        description = self.truncate_text(description, 6000)  # Increased from 4000
        
        # Store prompt length for metrics
        prompt_length = len(description)
        
        try:
            # Execute the task
            result = None
            
            # If we have a CrewAI agent, use it
            if self.agent:
                try:
                    # Create a Task object for CrewAI compatibility
                    from crewai import Task
                    task = Task(
                        description=description,
                        expected_output=f"Results of {self.agent_type} for {self.crew_type}"
                    )
                    
                    # Execute the task with the task object
                    result = self.agent.execute_task(task)
                except (TypeError, AttributeError) as e:
                    # If the above fails, try the older approach
                    logger.warning(f"Task object approach failed: {str(e)}. Trying direct string approach.")
                    try:
                        # Try with direct string - some CrewAI versions expect this
                        result = self.agent.execute_task(
                            description,
                            f"Results of {self.agent_type} for {self.crew_type}"
                        )
                    except Exception as inner_e:
                        logger.error(f"Direct string approach also failed: {str(inner_e)}")
                        raise
            else:
                # Use direct LLM call if no agent
                result = self.llm_client.generate_completion(description)
            
            # Calculate execution time
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Clean up and validate the result
            validated_result = self.validate_output(result)
            
            # Update execution stats
            self._update_execution_stats(execution_time, prompt_length, len(str(result)))
            
            # Add execution metadata if result is a dictionary
            if isinstance(validated_result, dict):
                validated_result["_metadata"] = {
                    "agent_type": self.agent_type,
                    "execution_id": execution_id,
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat(),
                    "prompt_length": prompt_length,
                    "result_length": len(str(result))
                }
            
            return validated_result
            
        except Exception as e:
            # Handle errors
            logger.error(f"Error executing task with {self.agent_type} agent: {str(e)}")
            
            # Calculate execution time even for errors
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            self._update_execution_stats(execution_time, prompt_length, 0, error=str(e))
            
            # Return error information instead of raising
            return {
                "error": str(e),
                "agent_type": self.agent_type,
                "_metadata": {
                    "error": True,
                    "execution_id": execution_id,
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    def _update_execution_stats(self, execution_time, prompt_length, result_length, error=None):
        """Update the agent's execution statistics."""
        self.execution_stats["runs"] += 1
        self.execution_stats["last_run_time"] = datetime.now().isoformat()
        self.execution_stats["total_execution_time"] += execution_time
        self.execution_stats["average_execution_time"] = (
            self.execution_stats["total_execution_time"] / self.execution_stats["runs"]
        )
        
        # Record metadata about this execution
        self.execution_stats["last_execution_metadata"] = {
            "execution_time": execution_time,
            "prompt_length": prompt_length,
            "result_length": result_length,
            "timestamp": datetime.now().isoformat(),
            "error": error
        }
        
        if self.verbose and self.execution_stats["runs"] % 5 == 0:
            logger.info(
                f"{self.agent_type.title()} agent stats: {self.execution_stats['runs']} runs, "
                f"avg time: {self.execution_stats['average_execution_time']:.2f}s"
            )
    
    def validate_output(self, output: Any) -> Any:
        """
        Validate and clean up agent output.
        
        Args:
            output: Raw output from agent
            
        Returns:
            Validated and cleaned output
        """
        # If output is a string, try to parse as JSON
        if isinstance(output, str):
            try:
                if output.strip().startswith('{') and output.strip().endswith('}'):
                    return json.loads(output)
            except json.JSONDecodeError:
                # Not valid JSON, return as is but log the issue
                if self.verbose:
                    logger.warning(f"{self.agent_type} output looks like JSON but couldn't be parsed")
                pass
        
        return output
    
    def truncate_text(self, text: str, max_length: int = None) -> str:
        """
        Truncate text to a maximum length to prevent large prompts.
        Enhanced version with smarter truncation strategy.
        
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
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics for this agent.
        
        Returns:
            Dictionary with execution statistics
        """
        return self.execution_stats
    
    def extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """
        Extract key keywords from text for better metadata.
        
        Args:
            text: Text to analyze
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of extracted keywords
        """
        # This is a simple keyword extraction implementation
        # Could be enhanced with NLP libraries in a real implementation
        
        # Get focus area keywords from config
        focus_area_keywords = []
        focus_areas = self.config.get("user_options", {}).get("focus_areas", {})
        for area, info in focus_areas.items():
            if "keywords" in info:
                focus_area_keywords.extend(info["keywords"])
        
        # Create a basic keyword frequency counter
        import re
        from collections import Counter
        
        # Tokenize and clean text
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stopwords (simplified list)
        stopwords = {"the", "and", "to", "of", "a", "in", "that", "it", "with", "for", "on", "is", "was", "be", "this", "are"}
        filtered_words = [word for word in words if word not in stopwords]
        
        # Count frequencies
        word_counts = Counter(filtered_words)
        
        # Prioritize focus area keywords that appear in the text
        prioritized_keywords = [word for word in focus_area_keywords if word in word_counts]
        
        # Add other frequent words up to max_keywords
        remaining_slots = max_keywords - len(prioritized_keywords)
        if remaining_slots > 0:
            other_keywords = [word for word, _ in word_counts.most_common(remaining_slots) 
                             if word not in prioritized_keywords]
            prioritized_keywords.extend(other_keywords)
        
        return prioritized_keywords[:max_keywords]