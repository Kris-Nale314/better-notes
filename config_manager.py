"""
Config Manager for Better Notes.
Simplified configuration management that works with the new agent architecture.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Manages configuration for the Better Notes system.
    Loads and provides access to configuration files.
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize the config manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.configs = {}
        
        # Ensure config directory exists
        if not self.config_dir.exists():
            logger.warning(f"Config directory '{config_dir}' not found. Using default configurations.")
        
        logger.info(f"ConfigManager initialized with config directory: {config_dir}")
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """
        Get configuration by name.
        
        Args:
            config_name: Name of the configuration to get
            
        Returns:
            Configuration dictionary
        """
        # Check if already loaded
        if config_name in self.configs:
            return self.configs[config_name]
        
        # Try to load the configuration
        config = self._load_config(config_name)
        
        # Cache for future use
        self.configs[config_name] = config
        
        return config
    
    def _load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_name: Name of the configuration to load
            
        Returns:
            Configuration dictionary
        """
        # Generate file path
        config_file = self.config_dir / f"{config_name}_config.json"
        
        try:
            # Try to load the file
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            # Try alternative location (agents/config)
            agent_config_path = Path("agents") / "config" / f"{config_name}_config.json"
            if agent_config_path.exists():
                with open(agent_config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
                    
            # If not found, return default config
            logger.warning(f"Config file for '{config_name}' not found. Using default.")
            return self._get_default_config(config_name)
            
        except Exception as e:
            logger.error(f"Error loading config '{config_name}': {e}")
            return self._get_default_config(config_name)
    
    def _get_default_config(self, config_name: str) -> Dict[str, Any]:
        """
        Get default configuration for a specific type.
        
        Args:
            config_name: Name of the configuration
            
        Returns:
            Default configuration dictionary
        """
        if config_name == "issues":
            return self._get_default_issues_config()
        else:
            # Generic default config
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
        Save configuration to file.
        
        Args:
            config_name: Name of the configuration
            config: Configuration dictionary
            
        Returns:
            True if successful, False otherwise
        """
        # Ensure config directory exists
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Generate file path
        config_file = self.config_dir / f"{config_name}_config.json"
        
        try:
            # Save the configuration
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            
            # Update cache
            self.configs[config_name] = config
            
            logger.info(f"Config '{config_name}' saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving config '{config_name}': {e}")
            return False