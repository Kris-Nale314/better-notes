"""
OrchestratorFactory - Creates properly configured orchestrator instances.
Simplifies the creation and configuration of orchestrators.
"""

import os
import logging
from typing import Dict, Any, Optional, Union

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
        try:
            # Create config manager if not provided
            if not config_manager:
                config_manager = ConfigManager()
            
            # Create or wrap LLM client
            if not isinstance(llm_client, UniversalLLMAdapter) and llm_client is not None:
                llm_client = UniversalLLMAdapter(
                    llm_client=llm_client,
                    api_key=api_key,
                    model=model,
                    temperature=temperature
                )
            elif llm_client is None:
                llm_client = UniversalLLMAdapter(
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
            
        except Exception as e:
            logger.error(f"Error creating orchestrator: {e}")
            raise
    
    @staticmethod
    def create_from_options(
        options: Union[ProcessingOptions, Dict[str, Any]],
        api_key: Optional[str] = None,
        llm_client = None,
        config_manager: Optional[ConfigManager] = None
    ) -> Orchestrator:
        """
        Create an orchestrator from ProcessingOptions.
        
        Args:
            options: Processing options or options dictionary
            api_key: Optional API key
            llm_client: Optional LLM client
            config_manager: Optional config manager
            
        Returns:
            Configured orchestrator instance
        """
        # Convert dictionary to ProcessingOptions if needed
        if isinstance(options, dict):
            if config_manager is None:
                config_manager = ConfigManager()
            options = config_manager.create_options_from_dict(options)
        
        return OrchestratorFactory.create_orchestrator(
            api_key=api_key,
            llm_client=llm_client,
            model=options.model_name,
            temperature=options.temperature,
            max_chunk_size=options.max_chunk_size,
            max_rpm=options.max_rpm,
            verbose=True,
            config_manager=config_manager
        )