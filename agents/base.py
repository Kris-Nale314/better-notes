# agents/base.py
from typing import Dict, Any, Optional, List
from crewai import Agent
import os
import json
import logging

logger = logging.getLogger(__name__)

class BaseAgent:
    """
    Base class for all specialized agents in Better Notes.
    Provides common functionality and configuration loading with optimizations
    for handling large documents.
    """
    
    def __init__(
        self,
        llm_client,
        agent_type: str,
        crew_type: str,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        max_chunk_size: int = 1500
    ):
        """
        Initialize a base agent with optimizations for large document handling.
        
        Args:
            llm_client: LLM client for agent communication
            agent_type: Type of agent (extraction, aggregation, evaluation, formatting)
            crew_type: Type of crew (issues, actions, opportunities)
            config: Optional pre-loaded configuration
            verbose: Whether to enable verbose mode
            max_chunk_size: Maximum size of text chunks to process
        """
        self.llm_client = llm_client
        self.agent_type = agent_type
        self.crew_type = crew_type
        self.verbose = verbose
        self.max_chunk_size = max_chunk_size
        
        # Load or use provided config
        self.config = config if config else self.load_config(crew_type)
        
        # Get agent-specific configuration
        agent_config = self.config.get("agent_config", {}).get(agent_type, {})
        if not agent_config:
            # Fallback to default values if specific config not found
            agent_config = self._get_default_agent_config(agent_type, crew_type)
        
        # Optimize backstory for large document handling
        backstory = agent_config.get("backstory", f"I am an expert in {crew_type} {agent_type}.")
        compact_backstory = backstory[:200] if len(backstory) > 200 else backstory
        
        # Create the CrewAI agent with optimized settings
        self.agent = Agent(
            role=agent_config.get("role", f"{crew_type.title()} {agent_type.title()} Agent"),
            goal=agent_config.get("goal", f"Process {crew_type} through {agent_type}"),
            backstory=compact_backstory,
            verbose=verbose,
            llm=llm_client,
            # Add specific settings to help with large contexts
            max_iterations=2,
            max_rpm=10,
            llm_config={
                "max_tokens": 600
            }
        )
    
    def load_config(self, crew_type: str) -> Dict[str, Any]:
        """
        Load a configuration file for the specified crew type.
        
        Args:
            crew_type: Type of crew (issues, actions, opportunities)
            
        Returns:
            Configuration dictionary
        """
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "agents", "config", f"{crew_type}_config.json"
        )
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {config_path}")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Error parsing configuration file: {config_path}")
            return {}
    
    def _get_default_agent_config(self, agent_type: str, crew_type: str) -> Dict[str, Any]:
        """
        Get default configuration for an agent if not specified in config file.
        
        Args:
            agent_type: Type of agent
            crew_type: Type of crew
            
        Returns:
            Default configuration dictionary
        """
        # Map of default roles, goals, and backstories
        defaults = {
            "extraction": {
                "issues": {
                    "role": "Issue Extractor",
                    "goal": "Identify all potential issues, problems, and challenges in documents",
                    "backstory": "I'm an expert at spotting problems and risks in documents."
                },
                "actions": {
                    "role": "Action Item Extractor",
                    "goal": "Extract all action items, tasks, and commitments from documents",
                    "backstory": "I specialize in identifying tasks and responsibilities in documents."
                },
                "insights": {
                    "role": "Insight Extractor",
                    "goal": "Discover key insights and context from documents",
                    "backstory": "I excel at understanding the essence of communications."
                }
            },
            "aggregation": {
                "role": f"{crew_type.title()} Aggregator",
                "goal": f"Combine and deduplicate {crew_type} from multiple extractions",
                "backstory": f"I specialize in organizing and consolidating {crew_type} information."
            },
            "evaluation": {
                "role": f"{crew_type.title()} Evaluator",
                "goal": f"Assess importance and impact of identified {crew_type}",
                "backstory": f"I'm an expert at prioritizing and evaluating {crew_type}."
            },
            "formatting": {
                "role": "Report Formatter",
                "goal": "Create clear, structured reports from analysis results",
                "backstory": "I specialize in presenting information in the most useful format."
            }
        }
        
        # Get agent-specific defaults
        if agent_type == "extraction":
            return defaults.get("extraction", {}).get(crew_type, {})
        else:
            return defaults.get(agent_type, {})
    
    def get_prompt_template(self) -> str:
        """
        Get the prompt template for this agent type.
        
        Returns:
            Prompt template string
        """
        return self.config.get(self.agent_type, {}).get("prompt_template", "")
    
    def get_output_schema(self) -> Dict[str, Any]:
        """
        Get the output schema for this agent type.
        
        Returns:
            Output schema dictionary
        """
        return self.config.get(self.agent_type, {}).get("output_schema", {})
    
    def get_format_template(self) -> str:
        """
        Get the format template for formatting agents.
        
        Returns:
            Format template string
        """
        if self.agent_type == "formatting":
            return self.config.get("formatting", {}).get("format_template", "")
        return ""
    
    def build_prompt(self, context: Dict[str, Any] = None) -> str:
        """
        Build a prompt using the template and context.
        
        Args:
            context: Dictionary of context variables to fill the template
            
        Returns:
            Formatted prompt string
        """
        template = self.get_prompt_template()
        context = context or {}
        
        # Simple template substitution
        for key, value in context.items():
            placeholder = f"{{{key}}}"
            if isinstance(value, str):
                value = self.truncate_text(value, 2000)  # Ensure context values aren't too large
                template = template.replace(placeholder, value)
            elif isinstance(value, (list, dict)):
                # Truncate JSON representations to prevent oversized headers
                json_str = json.dumps(value, indent=None)[:1000]
                template = template.replace(placeholder, json_str)
            else:
                template = template.replace(placeholder, str(value))
        
        # Ensure final prompt isn't too large
        return self.truncate_text(template, 4000)
    
    def execute_task(self, description: str = None, context: Dict[str, Any] = None) -> Any:
        """
        Execute a task using this agent.
        
        Args:
            description: Optional task description (overrides built prompt)
            context: Context for building the prompt if description not provided
            
        Returns:
            Task result
        """
        if not description and context:
            description = self.build_prompt(context)
        elif not description:
            description = self.build_prompt()
        
        # Ensure description isn't too large
        description = self.truncate_text(description, 4000)
        
        return self.agent.execute_task(
            description=description,
            expected_output=f"Results of {self.agent_type} for {self.crew_type}"
        )
    
    def truncate_text(self, text: str, max_length: int = None) -> str:
        """
        Truncate text to a maximum length to prevent large headers.
        
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
            
        return text[:max_length] + "..."