"""
OrchestratorFactory - Creates properly configured orchestrator instances.
Simplifies the creation and configuration of orchestrators.
"""

import logging
from typing import Dict, Any, Optional

from universal_llm_adapter import UniversalLLMAdapter
from config_manager import ConfigManager, ProcessingOptions
from orchestrator import Orchestrator

logger = logging.getLogger(__name__)

class OrchestratorFactory:
    """Factory for creating properly configured orchestrator instances."""
    
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
        # Create or get config manager
        if config_manager is None:
            config_manager = ConfigManager()
        
        # Create or wrap LLM client
        if llm_client is None:
            llm_client = UniversalLLMAdapter(
                api_key=api_key,
                model=model,
                temperature=temperature
            )
        elif not isinstance(llm_client, UniversalLLMAdapter):
            llm_client = UniversalLLMAdapter(
                llm_client=llm_client,
                api_key=api_key,
                model=model,
                temperature=temperature
            )
        
        # Create orchestrator
        orchestrator = Orchestrator(
            llm_client=llm_client,
            verbose=verbose,
            max_chunk_size=max_chunk_size,
            max_rpm=max_rpm,
            config_manager=config_manager
        )
        
        return orchestrator
    
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
        # Create orchestrator with basic settings
        return OrchestratorFactory.create_orchestrator(
            api_key=api_key,
            llm_client=llm_client,
            model=options.get("model_name", "gpt-3.5-turbo"),
            temperature=options.get("temperature", 0.2),
            max_chunk_size=options.get("max_chunk_size", 10000),
            max_rpm=options.get("max_rpm", 10),
            verbose=True,
            config_manager=config_manager
        )