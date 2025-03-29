"""
Enhanced Config Manager for Better Notes.
Improved configuration loading with better error handling and validation.
"""

import os
import json
import logging
import traceback
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Enhanced configuration manager with improved loading, validation, and error handling.
    Manages configuration for the Better Notes system.
    """
    
    # Singleton instance
    _instance = None
    
    @classmethod
    def get_instance(cls, config_dir: str = "config"):
        """
        Get or create the singleton instance.
        
        Args:
            config_dir: Configuration directory
            
        Returns:
            ConfigManager instance
        """
        if cls._instance is None:
            cls._instance = ConfigManager(config_dir)
        return cls._instance
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize the config manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.configs = {}
        self.default_configs = {}
        
        # Ensure config directory exists
        if not self.config_dir.exists():
            logger.warning(f"Config directory '{config_dir}' not found. Creating it.")
            self.config_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ConfigManager initialized with config directory: {config_dir}")
        
        # Preload default configurations
        self._init_default_configs()
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """
        Get configuration by name with improved error handling.
        
        Args:
            config_name: Name of the configuration to get
            
        Returns:
            Configuration dictionary
        """
        logger.info(f"Requested config: {config_name}")
        
        # Check if already loaded
        if config_name in self.configs:
            logger.info(f"Returning cached config for: {config_name}")
            return self.configs[config_name]
        
        # Try multiple locations with detailed logging
        config = None
        locations_tried = []
        
        # Try main config directory
        main_path = self.config_dir / f"{config_name}_config.json"
        locations_tried.append(str(main_path))
        if main_path.exists():
            logger.info(f"Found config at: {main_path}")
            config = self._load_file(main_path)
        
        # Try agents/config directory
        if config is None:
            agent_path = Path("agents") / "config" / f"{config_name}_config.json"
            locations_tried.append(str(agent_path))
            if agent_path.exists():
                logger.info(f"Found config at: {agent_path}")
                config = self._load_file(agent_path)
        
        # Try root config directory
        if config is None:
            root_path = Path("config") / f"{config_name}_config.json"
            locations_tried.append(str(root_path))
            if root_path.exists():
                logger.info(f"Found config at: {root_path}")
                config = self._load_file(root_path)
        
        # Use default if not found
        if config is None:
            logger.warning(f"Config '{config_name}' not found at any location: {', '.join(locations_tried)}")
            logger.info(f"Using default config for: {config_name}")
            config = self._get_default_config(config_name)
        
        # Validate the config
        if not self._validate_config(config, config_name):
            logger.warning(f"Invalid config for '{config_name}', using default")
            config = self._get_default_config(config_name)
        
        # Cache for future use
        self.configs[config_name] = config
        
        # Log success
        logger.info(f"Successfully loaded config for '{config_name}' with {len(config)} keys")
        
        return config
    
    def _load_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load configuration from file with error handling.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            Configuration dictionary or None if loading failed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Successfully loaded config from: {file_path}")
            return config
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in {file_path}: {e}")
            logger.error(f"Error at line {e.lineno}, column {e.colno}: {e.msg}")
            return None
        except Exception as e:
            logger.error(f"Error loading config from {file_path}: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def _validate_config(self, config: Dict[str, Any], config_name: str) -> bool:
        """
        Validate configuration structure.
        
        Args:
            config: Configuration to validate
            config_name: Configuration name
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(config, dict):
            logger.error(f"Config for '{config_name}' is not a dictionary")
            return False
        
        # Check for required keys based on config type
        if config_name == "issues":
            required_keys = ["crew_type", "workflow"]
            if not all(key in config for key in required_keys):
                missing = [key for key in required_keys if key not in config]
                logger.error(f"Missing required keys in issues config: {missing}")
                return False
            
            # Check for agent roles
            if "workflow" in config and "agent_roles" not in config["workflow"]:
                logger.error("Missing 'agent_roles' in workflow config")
                return False
        
        return True
    
    def _init_default_configs(self):
        """Initialize default configurations."""
        # Add default issues config
        self.default_configs["issues"] = self._get_default_issues_config()
    
    def _get_default_config(self, config_name: str) -> Dict[str, Any]:
        """
        Get default configuration for a specific type.
        
        Args:
            config_name: Name of the configuration
            
        Returns:
            Default configuration dictionary
        """
        # Return cached default if available
        if config_name in self.default_configs:
            return self.default_configs[config_name]
        
        # Generic default config for unknown types
        return {
            "crew_type": config_name,
            "description": f"Default configuration for {config_name}",
            "workflow": {
                "enabled_stages": ["document_analysis", "chunking", "planning", 
                                "extraction", "aggregation", "evaluation", 
                                "formatting", "review"],
                "agent_roles": {
                    "planner": {
                        "description": "Plans the analysis approach",
                        "primary_task": "Create tailored instructions for each agent"
                    },
                    "extractor": {
                        "description": "Extracts relevant information from chunks",
                        "primary_task": f"Extract {config_name} from each document chunk"
                    },
                    "aggregator": {
                        "description": "Combines and deduplicates extraction results",
                        "primary_task": f"Combine similar {config_name} from all document chunks"
                    },
                    "evaluator": {
                        "description": "Evaluates importance and priority",
                        "primary_task": f"Evaluate each {config_name} for importance and impact"
                    },
                    "formatter": {
                        "description": "Creates a structured report",
                        "primary_task": f"Format {config_name} into a clear, organized report"
                    },
                    "reviewer": {
                        "description": "Reviews report quality",
                        "primary_task": "Review the report for quality and alignment with user needs"
                    }
                }
            }
        }
    
    def _get_default_issues_config(self) -> Dict[str, Any]:
        """
        Get default configuration for issues analysis.
        
        Returns:
            Default issues configuration
        """
        return {
            "crew_type": "issues",
            "description": "Identifies problems, challenges, risks, and concerns in documents",
            
            "issue_definition": {
                "description": "Any problem, challenge, risk, or concern that may impact objectives, efficiency, or quality",
                "severity_levels": {
                    "critical": "Immediate threat requiring urgent attention",
                    "high": "Significant impact requiring prompt attention",
                    "medium": "Moderate impact that should be addressed",
                    "low": "Minor impact with limited consequences"
                },
                "categories": [
                    "technical", "process", "resource", "quality", "risk", "compliance"
                ]
            },
            
            "workflow": {
                "enabled_stages": ["document_analysis", "chunking", "planning", 
                                "extraction", "aggregation", "evaluation", 
                                "formatting", "review"],
                "agent_roles": {
                    "planner": {
                        "description": "Plans the analysis approach",
                        "primary_task": "Create tailored instructions for each agent based on document type and user preferences"
                    },
                    "extractor": {
                        "description": "Identifies issues from document chunks",
                        "primary_task": "Find all issues, assign initial severity, and provide relevant context",
                        "output_schema": {
                            "title": "Concise issue label",
                            "description": "Detailed explanation of the issue",
                            "severity": "Initial severity assessment (critical/high/medium/low)",
                            "category": "Issue category from the defined list",
                            "context": "Relevant information from the document"
                        }
                    },
                    "aggregator": {
                        "description": "Combines and deduplicates issues from all chunks",
                        "primary_task": "Consolidate similar issues while preserving important distinctions"
                    },
                    "evaluator": {
                        "description": "Assesses issue severity and priority",
                        "primary_task": "Analyze each issue's impact and assign final severity and priority"
                    },
                    "formatter": {
                        "description": "Creates the structured report",
                        "primary_task": "Organize issues by severity and category into a clear report"
                    },
                    "reviewer": {
                        "description": "Ensures quality and alignment with user needs",
                        "primary_task": "Verify report quality and alignment with user preferences",
                        "review_criteria": {
                            "alignment": "Does the analysis align with user instructions and focus areas?",
                            "completeness": "Does the report address all significant issues at the appropriate detail level?",
                            "consistency": "Are severity ratings applied consistently throughout the analysis?",
                            "clarity": "Is the report clear, well-organized, and actionable?",
                            "balance": "Are issues presented in a balanced way without over or under-emphasis?"
                        }
                    }
                }
            },
            
            "user_options": {
                "detail_levels": {
                    "essential": "Focus only on the most significant issues",
                    "standard": "Balanced analysis of important issues",
                    "comprehensive": "In-depth analysis of all potential issues"
                },
                "focus_areas": {
                    "technical": "Implementation, architecture, technology issues",
                    "process": "Workflow, procedure, methodology issues",
                    "resource": "Staffing, budget, time, materials constraints",
                    "quality": "Standards, testing, performance concerns",
                    "risk": "Compliance, security, strategic risks"
                }
            },
            
            "report_format": {
                "sections": [
                    "Executive Summary",
                    "Critical Issues",
                    "High-Priority Issues", 
                    "Medium-Priority Issues",
                    "Low-Priority Issues"
                ],
                "issue_presentation": {
                    "title": "Clear, descriptive title",
                    "severity": "Visual indicator of severity",
                    "description": "Full issue description",
                    "impact": "Potential consequences",
                    "category": "Issue category"
                }
            }
        }
    
    def save_config(self, config_name: str, config: Dict[str, Any]) -> bool:
        """
        Save configuration to file with error handling.
        
        Args:
            config_name: Name of the configuration
            config: Configuration dictionary
            
        Returns:
            True if successful, False otherwise
        """
        # Validate config before saving
        if not self._validate_config(config, config_name):
            logger.error(f"Cannot save invalid config for '{config_name}'")
            return False
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate file path
        config_file = self.config_dir / f"{config_name}_config.json"
        
        try:
            # Save the configuration with pretty formatting
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            # Update cache
            self.configs[config_name] = config
            
            logger.info(f"Config '{config_name}' saved successfully to {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving config '{config_name}': {e}")
            logger.error(traceback.format_exc())
            return False
    
    def list_configs(self) -> List[str]:
        """
        List available configuration names.
        
        Returns:
            List of configuration names
        """
        config_files = []
        
        # Check main config directory
        if self.config_dir.exists():
            config_files.extend(self.config_dir.glob("*_config.json"))
        
        # Check agents/config directory
        agents_config_dir = Path("agents") / "config"
        if agents_config_dir.exists():
            config_files.extend(agents_config_dir.glob("*_config.json"))
        
        # Extract config names (remove _config.json suffix)
        config_names = [f.stem.replace("_config", "") for f in config_files]
        
        # Add default configs not found in files
        for name in self.default_configs:
            if name not in config_names:
                config_names.append(name)
        
        return sorted(set(config_names))
    
    def create_config_ui(self, config_name: str) -> Dict[str, Any]:
        """
        Create a UI-friendly version of a configuration.
        
        Args:
            config_name: Name of the configuration
            
        Returns:
            UI-friendly configuration dictionary
        """
        # Get the configuration
        config = self.get_config(config_name)
        
        # Create UI-friendly version
        ui_config = {
            "name": config_name,
            "description": config.get("description", f"Configuration for {config_name}"),
            "sections": []
        }
        
        # Basic settings section
        basic_section = {
            "name": "Basic Settings",
            "settings": [
                {
                    "key": "crew_type",
                    "label": "Crew Type",
                    "type": "text",
                    "value": config.get("crew_type", config_name),
                    "description": "Type of analysis crew"
                },
                {
                    "key": "description",
                    "label": "Description",
                    "type": "textarea",
                    "value": config.get("description", ""),
                    "description": "Description of this configuration"
                }
            ]
        }
        ui_config["sections"].append(basic_section)
        
        # Workflow section
        workflow = config.get("workflow", {})
        workflow_section = {
            "name": "Workflow",
            "settings": [
                {
                    "key": "enabled_stages",
                    "label": "Enabled Stages",
                    "type": "multiselect",
                    "value": workflow.get("enabled_stages", []),
                    "options": ["document_analysis", "chunking", "planning", 
                               "extraction", "aggregation", "evaluation", 
                               "formatting", "review"],
                    "description": "Processing stages to enable"
                }
            ]
        }
        ui_config["sections"].append(workflow_section)
        
        # Agent roles section
        agent_roles = workflow.get("agent_roles", {})
        for agent_type in ["planner", "extractor", "aggregator", "evaluator", "formatter", "reviewer"]:
            if agent_type in agent_roles:
                role = agent_roles[agent_type]
                agent_section = {
                    "name": f"{agent_type.capitalize()} Agent",
                    "settings": [
                        {
                            "key": f"agent_roles.{agent_type}.description",
                            "label": "Description",
                            "type": "text",
                            "value": role.get("description", ""),
                            "description": f"Description of the {agent_type} agent's role"
                        },
                        {
                            "key": f"agent_roles.{agent_type}.primary_task",
                            "label": "Primary Task",
                            "type": "textarea",
                            "value": role.get("primary_task", ""),
                            "description": f"Main task for the {agent_type} agent"
                        }
                    ]
                }
                
                # Add output schema for extractor
                if agent_type == "extractor" and "output_schema" in role:
                    output_schema = role["output_schema"]
                    for field, desc in output_schema.items():
                        agent_section["settings"].append({
                            "key": f"agent_roles.{agent_type}.output_schema.{field}",
                            "label": f"Schema Field: {field}",
                            "type": "text",
                            "value": desc,
                            "description": f"Description for {field} in the output schema"
                        })
                
                ui_config["sections"].append(agent_section)
        
        return ui_config