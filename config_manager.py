"""
ConfigManager - Integrated configuration and processing options management.
Provides a centralized way to handle configurations across different environments.
Fixed to properly use dataclasses and handle fields correctly.
"""

import os
import json
import logging
import dataclasses
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ProcessingOptions:
    """
    Configuration options for document processing and analysis.
    Combines the functionality previously in options.py.
    """
    # --- Core Model Settings ---
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.2

    # --- Chunking Settings ---
    min_chunks: int = 3
    max_chunk_size: Optional[int] = None  # Auto-calculated if None

    # --- Detail Level ---
    detail_level: str = "standard"  # Options: "essential", "standard", "comprehensive"

    # --- Analysis Settings ---
    preview_length: int = 2000  # Characters for document preview analysis

    # --- Performance Settings ---
    max_concurrent_chunks: int = 5
    max_rpm: int = 10
    enable_caching: bool = True
    cache_dir: str = ".cache"

    # --- Output Settings ---
    include_metadata: bool = True

    # --- User Instructions ---
    user_instructions: Optional[str] = None

    # --- Focus Areas ---
    focus_areas: List[str] = field(default_factory=list)
    
    # --- Analysis Types ---
    crews: List[str] = field(default_factory=lambda: ["issues"])
    
    # --- Reviewer ---
    enable_reviewer: bool = True
    
    # --- Pass-Specific Options ---
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
    """Centralized configuration management with fallbacks."""
    
    def __init__(self, base_dir=None):
        """
        Initialize config manager with optional base directory.
        
        Args:
            base_dir: Base directory for configuration files (defaults to parent of current dir)
        """
        self.base_dir = base_dir
        if not self.base_dir:
            # Try to determine base directory automatically
            try:
                # First try to use current file's location
                self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            except:
                # If that fails, use current working directory
                self.base_dir = os.getcwd()
                
        self.config_cache = {}
        logger.info(f"ConfigManager initialized with base_dir: {self.base_dir}")
        
        # Ensure cache directory exists
        cache_dir = os.path.join(self.base_dir, ".cache")
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """
        Get configuration with fallbacks to multiple locations.
        
        Args:
            config_name: Name of the configuration (without _config.json suffix)
            
        Returns:
            Dictionary with configuration, or empty dict if not found
        """
        # Return cached config if available
        if config_name in self.config_cache:
            return self.config_cache[config_name]
        
        # Try multiple possible locations
        possible_paths = self._get_possible_paths(config_name)
        
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        logger.info(f"Loaded configuration from {path}")
                        self.config_cache[config_name] = config
                        return config
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.debug(f"Could not load config from {path}: {e}")
                continue
        
        # If no config found, log warning and return empty dict
        logger.warning(f"Could not find configuration for {config_name}")
        return {}
    
    def _get_possible_paths(self, config_name: str) -> List[str]:
        """
        Get a list of possible paths for a configuration file.
        
        Args:
            config_name: Name of the configuration
            
        Returns:
            List of possible file paths
        """
        filename = f"{config_name}_config.json"
        
        return [
            # Current directory
            os.path.join(".", filename),
            
            # Config in standard locations
            os.path.join(self.base_dir, "agents", "config", filename),
            os.path.join(self.base_dir, "config", filename),
            
            # Config in parent directory
            os.path.join(os.path.dirname(self.base_dir), "agents", "config", filename),
            
            # Relative paths
            os.path.join("agents", "config", filename),
            os.path.join("config", filename),
            
            # Additional fallbacks
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", filename),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
        ]
    
    def save_config(self, config_name: str, config_data: Dict[str, Any]) -> Optional[str]:
        """
        Save a configuration to file.
        
        Args:
            config_name: Name of the configuration
            config_data: Configuration data to save
            
        Returns:
            Path where configuration was saved, or None if save failed
        """
        # Create config dir if not exists
        try:
            config_dir = os.path.join(self.base_dir, "config")
            os.makedirs(config_dir, exist_ok=True)
            
            # Save to file
            filename = f"{config_name}_config.json"
            path = os.path.join(config_dir, filename)
            
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
        try:
            options_dict = options.to_dict()
            path = self.save_config("processing_options", options_dict)
            return path is not None
        except Exception as e:
            logger.error(f"Error saving processing options: {e}")
            return False
    
    def create_options_from_dict(self, options_dict: Dict[str, Any]) -> ProcessingOptions:
        """
        Create ProcessingOptions from a dictionary.
        
        Args:
            options_dict: Dictionary with options
            
        Returns:
            ProcessingOptions object
        """
        return ProcessingOptions.from_dict(options_dict)

    def load_config_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load configuration from a specific file path.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            Configuration dictionary or empty dict if loading failed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"Loaded configuration from {file_path}")
                return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading config from {file_path}: {e}")
            return {}

    def get_config_path(self, config_name: str) -> Optional[str]:
        """
        Get the path to a configuration file.
        
        Args:
            config_name: Name of the configuration
            
        Returns:
            Path to the configuration file or None if not found
        """
        possible_paths = self._get_possible_paths(config_name)
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        return None

    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all cached configurations.
        
        Returns:
            Dictionary of all cached configurations
        """
        return self.config_cache

    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self.config_cache = {}
        logger.info("Configuration cache cleared")