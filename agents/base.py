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
    Provides common functionality and configuration loading.
    """
    
    def __init__(
        self,
        llm_client,
        agent_type: str,
        crew_type: str,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ):
        """
        Initialize a base agent.
        
        Args:
            llm_client: LLM client for agent communication
            agent_type: Type of agent (extraction, aggregation, evaluation, formatting)
            crew_type: Type of crew (issues, actions, opportunities)
            config: Optional pre-loaded configuration
            verbose: Whether to enable verbose mode
        """
        self.llm_client = llm_client
        self.agent_type = agent_type
        self.crew_type = crew_type
        self.verbose = verbose
        
        # Load or use provided config
        self.config = config if config else self.load_config(crew_type)
        
        # Get agent-specific configuration
        agent_config = self.config.get("agent_config", {}).get(agent_type, {})
        if not agent_config:
            # Fallback to default values if specific config not found
            agent_config = self._get_default_agent_config(agent_type, crew_type)
        
        # Create the CrewAI agent
        self.agent = Agent(
            role=agent_config.get("role", f"{crew_type.title()} {agent_type.title()} Agent"),
            goal=agent_config.get("goal", f"Process {crew_type} through {agent_type}"),
            backstory=agent_config.get("backstory", f"I am an expert in {crew_type} {agent_type}."),
            verbose=verbose,
            llm=llm_client
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
                "opportunities": {
                    "role": "Opportunity Identifier",
                    "goal": "Discover potential improvements and opportunities in documents",
                    "backstory": "I excel at finding opportunities where others see challenges."
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
                template = template.replace(placeholder, value)
            elif isinstance(value, (list, dict)):
                template = template.replace(placeholder, json.dumps(value, indent=2))
            else:
                template = template.replace(placeholder, str(value))
        
        return template
    
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
        
        return self.agent.execute_task(
            description=description,
            expected_output=f"Results of {self.agent_type} for {self.crew_type}"
        )