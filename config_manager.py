"""
ConfigManager - Simplified configuration and processing options management.
Provides a centralized way to handle configurations for Better Notes.
"""

import os
import json
import logging
import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

@dataclass
class ProcessingOptions:
    """
    Configuration options for document processing and analysis.
    """
    # Core Model Settings
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.2

    # Chunking Settings
    min_chunks: int = 3
    max_chunk_size: Optional[int] = None

    # Detail Level
    detail_level: str = "standard"  # essential, standard, comprehensive

    # Analysis Settings
    preview_length: int = 2000

    # Performance Settings
    max_concurrent_chunks: int = 5
    max_rpm: int = 10
    enable_caching: bool = True
    cache_dir: str = ".cache"

    # Output Settings
    include_metadata: bool = True

    # User Instructions
    user_instructions: Optional[str] = None

    # Focus Areas
    focus_areas: List[str] = field(default_factory=list)
    
    # Analysis Types
    crews: List[str] = field(default_factory=lambda: ["issues"])
    
    # Reviewer
    enable_reviewer: bool = True
    
    # Pass-Specific Options
    pass_options: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert options to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingOptions':
        """Create options from dictionary."""
        # Get valid field names using dataclasses.fields()
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        
        # Filter out keys that aren't in the dataclass
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)


class ConfigManager:
    """Simple configuration management for Better Notes."""
    
    def __init__(self, base_dir=None):
        """
        Initialize the config manager.
        
        Args:
            base_dir: Base directory for configuration files
        """
        # Use provided base_dir or determine from current directory
        self.base_dir = base_dir or os.getcwd()
        self.config_cache = {}
        
        logger.info(f"ConfigManager initialized with base_dir: {self.base_dir}")
        
        # Ensure cache directory exists
        os.makedirs(os.path.join(self.base_dir, ".cache"), exist_ok=True)
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """
        Get configuration by name.
        
        Args:
            config_name: Name of the configuration (without _config.json suffix)
            
        Returns:
            Configuration dictionary or empty dict if not found
        """
        # Return cached config if available
        if config_name in self.config_cache:
            return self.config_cache[config_name]
        
        # Standard config locations to try
        config_paths = [
            os.path.join(self.base_dir, "config", f"{config_name}_config.json"),
            os.path.join(self.base_dir, "agents", "config", f"{config_name}_config.json"),
            os.path.join(self.base_dir, f"{config_name}_config.json"),
            os.path.join("config", f"{config_name}_config.json"),
            os.path.join("agents", "config", f"{config_name}_config.json"),
        ]
        
        # Try each path
        for path in config_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        self.config_cache[config_name] = config
                        logger.info(f"Loaded configuration from {path}")
                        return config
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in config file: {path}")
        
        # Log warning if no config found
        logger.warning(f"Could not find configuration for {config_name}")
        return {}
    
    def save_config(self, config_name: str, config_data: Dict[str, Any]) -> Optional[str]:
        """
        Save a configuration to file.
        
        Args:
            config_name: Name of the configuration
            config_data: Configuration data to save
            
        Returns:
            Path where configuration was saved, or None if save failed
        """
        # Create config directory if needed
        config_dir = os.path.join(self.base_dir, "config")
        os.makedirs(config_dir, exist_ok=True)
        
        # Save to file
        filename = f"{config_name}_config.json"
        path = os.path.join(config_dir, filename)
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2)
            
            # Update cache
            self.config_cache[config_name] = config_data
            logger.info(f"Saved configuration to {path}")
            return path
        except Exception as e:
            logger.error(f"Error saving configuration {config_name}: {e}")
            return None
    
    def get_processing_options(self) -> ProcessingOptions:
        """
        Get processing options from configuration or defaults.
        
        Returns:
            ProcessingOptions object
        """
        # Try to load options from config
        options_config = self.get_config("processing_options")
        
        # If config exists, create from dict
        if options_config:
            try:
                return ProcessingOptions.from_dict(options_config)
            except Exception as e:
                logger.error(f"Error creating ProcessingOptions from config: {e}")
        
        # Return default options
        return ProcessingOptions()
    
    def save_processing_options(self, options: ProcessingOptions) -> bool:
        """
        Save processing options to configuration.
        
        Args:
            options: ProcessingOptions object
            
        Returns:
            True if successful, False otherwise
        """
        options_dict = options.to_dict()
        path = self.save_config("processing_options", options_dict)
        return path is not None
    
    def create_options_from_dict(self, options_dict: Dict[str, Any]) -> ProcessingOptions:
        """
        Create ProcessingOptions from a dictionary.
        
        Args:
            options_dict: Dictionary with options
            
        Returns:
            ProcessingOptions object
        """
        return ProcessingOptions.from_dict(options_dict)