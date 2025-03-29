"""
OrchestratorFactory - Creates optimized orchestrator instances.
Provides access to both standard and LangChain implementations.
"""

import logging
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class OrchestratorFactory:
    """Factory for creating orchestrator instances with appropriate configuration."""
    
    @staticmethod
    def create_orchestrator(
        implementation: str = "langchain",
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.2,
        max_chunk_size: int = 10000,
        max_rpm: int = 10,
        verbose: bool = True,
        config_manager = None,
        **kwargs
    ) -> Any:
        """
        Create a configured orchestrator instance.
        
        Args:
            implementation: Orchestrator implementation ("langchain" or "standard")
            api_key: API key for LLM
            model: Model name
            temperature: Temperature for generation
            max_chunk_size: Maximum chunk size
            max_rpm: Maximum requests per minute
            verbose: Whether to enable verbose logging
            config_manager: Optional config manager instance
            **kwargs: Additional implementation-specific arguments
            
        Returns:
            Configured orchestrator instance
        """
        # Get config manager if not provided
        if config_manager is None:
            from config_manager import ConfigManager
            config_manager = ConfigManager.get_instance()
        
        # Common parameters for both implementations
        common_params = {
            "api_key": api_key,
            "model": model,
            "temperature": temperature,
            "max_chunk_size": max_chunk_size,
            "max_rpm": max_rpm,
            "verbose": verbose,
            "config_manager": config_manager,
            **kwargs
        }
        
        # Create the appropriate orchestrator based on implementation
        if implementation == "langchain":
            try:
                from orchestrator_langchain import LangChainOrchestrator
                logger.info(f"Creating LangChainOrchestrator with model: {model}")
                return LangChainOrchestrator(**common_params)
            except ImportError as e:
                logger.error(f"Failed to import LangChainOrchestrator: {e}")
                # Fall back to standard implementation
                logger.warning("Falling back to standard Orchestrator")
                implementation = "standard"
        
       
        
        # Unknown implementation
        raise ValueError(f"Unknown orchestrator implementation: {implementation}")
    
    @staticmethod
    def create_from_options(
        options: Dict[str, Any],
        implementation: str = "langchain", 
        api_key: Optional[str] = None,
        config_manager = None
    ) -> Any:
        """
        Create an orchestrator from an options dictionary.
        
        Args:
            options: Options dictionary with configuration
            implementation: Orchestrator implementation
            api_key: Optional API key
            config_manager: Optional config manager
            
        Returns:
            Configured orchestrator instance
        """
        # Extract relevant options
        orchestrator_params = {
            "model": options.get("model_name", options.get("model", "gpt-3.5-turbo")),
            "temperature": options.get("temperature", 0.2),
            "max_chunk_size": options.get("max_chunk_size", 10000),
            "max_rpm": options.get("max_rpm", 10),
            "verbose": options.get("verbose", True),
        }
        
        # Create the orchestrator
        return OrchestratorFactory.create_orchestrator(
            implementation=implementation,
            api_key=api_key,
            config_manager=config_manager,
            **orchestrator_params
        )
    
    @staticmethod
    def get_implementation_options() -> Dict[str, str]:
        """
        Get available orchestrator implementations with descriptions.
        
        Returns:
            Dictionary of implementation names and descriptions
        """
        return {
            "langchain": "LangChain-based implementation (recommended)",
            "standard": "Original CrewAI implementation"
        }