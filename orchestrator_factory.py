"""
OrchestratorFactory - Creates properly configured orchestrator instances.
Simplified implementation that works with the new agent architecture.
"""

import logging
from typing import Dict, Any, Optional

from universal_llm_adapter import LLMAdapter
from config_manager import ConfigManager
from orchestrator import Orchestrator

logger = logging.getLogger(__name__)

class OrchestratorFactory:
    """
    Factory for creating properly configured orchestrator instances.
    Supports the new agent architecture while maintaining backwards compatibility.
    """
    
    @staticmethod
    def create_orchestrator(
        api_key: Optional[str] = None,
        llm_client = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.2,
        max_chunk_size: int = 10000,
        verbose: bool = True,
        max_rpm: int = 10,
        config_manager: Optional[ConfigManager] = None
    ) -> Orchestrator:
        """
        Create a configured orchestrator instance.
        
        Args:
            api_key: OpenAI API key (can be None if llm_client is provided)
            llm_client: Existing LLM client (if None, one will be created)
            model: Model name for LLM
            temperature: Temperature for LLM
            max_chunk_size: Maximum chunk size
            verbose: Whether to enable verbose logging
            max_rpm: Maximum requests per minute
            config_manager: Optional config manager
            
        Returns:
            Configured orchestrator instance
        """
        logger.info(f"Creating orchestrator with model: {model}")
        
        # Create or use config manager
        if config_manager is None:
            config_manager = ConfigManager()
        
        # Create the orchestrator with parameters
        return Orchestrator(
            api_key=api_key,
            llm_client=llm_client,
            model=model,
            temperature=temperature,
            verbose=verbose,
            max_chunk_size=max_chunk_size,
            max_rpm=max_rpm,
            config_manager=config_manager
        )
    
    @staticmethod
    def create_from_options(
        options: Dict[str, Any],
        api_key: Optional[str] = None,
        llm_client = None,
        config_manager: Optional[ConfigManager] = None
    ) -> Orchestrator:
        """
        Create an orchestrator from options dictionary.
        
        Args:
            options: Options dictionary
            api_key: Optional API key
            llm_client: Optional LLM client
            config_manager: Optional config manager
            
        Returns:
            Configured orchestrator instance
        """
        # Extract options with defaults
        model = options.get("model_name", "gpt-3.5-turbo")
        temperature = options.get("temperature", 0.2)
        max_chunk_size = options.get("max_chunk_size", 10000)
        max_rpm = options.get("max_rpm", 10)
        verbose = options.get("verbose", True)
        
        # Create orchestrator with extracted options
        return OrchestratorFactory.create_orchestrator(
            api_key=api_key,
            llm_client=llm_client,
            model=model,
            temperature=temperature,
            max_chunk_size=max_chunk_size,
            max_rpm=max_rpm,
            verbose=verbose,
            config_manager=config_manager
        )
    
    @staticmethod
    def create_issues_crew(
        llm_client,
        config_manager: Optional[ConfigManager] = None,
        verbose: bool = True,
        max_chunk_size: int = 10000,
        max_rpm: int = 10
    ):
        """
        Create an IssuesCrew instance directly.
        
        Args:
            llm_client: LLM client for the crew
            config_manager: Optional config manager
            verbose: Whether to enable verbose logging
            max_chunk_size: Maximum chunk size
            max_rpm: Maximum requests per minute
            
        Returns:
            IssuesCrew instance
        """
        from crews.issues_crew import IssuesCrew
        
        # Create or use config manager
        if config_manager is None:
            config_manager = ConfigManager()
        
        # Create and return the issues crew
        return IssuesCrew(
            llm_client=llm_client,
            verbose=verbose,
            max_chunk_size=max_chunk_size,
            max_rpm=max_rpm,
            config_manager=config_manager
        )