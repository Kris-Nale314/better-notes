"""
Enhanced Planner Agent for Better Notes with improved error handling and JSON parsing.
Creates optimized, document-aware instructions for specialized agents.
"""

import json
import logging
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base import BaseAgent

logger = logging.getLogger(__name__)

class PlannerAgent(BaseAgent):
    """
    Enhanced Planner agent that creates tailored plans for document analysis.
    Features improved error handling, JSON parsing, and logging.
    """
    
    def __init__(
        self,
        llm_client,
        crew_type: str,
        config: Optional[Dict[str, Any]] = None,
        config_manager = None,
        verbose: bool = True,
        max_chunk_size: int = 2000,
        max_rpm: int = 10
    ):
        """Initialize a planner agent."""
        super().__init__(
            llm_client=llm_client,
            agent_type="planner",
            crew_type=crew_type,
            config=config,
            config_manager=config_manager,
            verbose=verbose,
            max_chunk_size=max_chunk_size,
            max_rpm=max_rpm
        )
        
        logger.info(f"PlannerAgent initialized for {crew_type}")
    
    async def process(self, context):
        """
        Create a plan using the provided context.
        
        Args:
            context: ProcessingContext object
            
        Returns:
            Planning result with instructions for each agent
        """
        logger.info("PlannerAgent starting planning process")
        
        try:
            # Execute planning
            document_info = getattr(context, 'document_info', {})
            options = getattr(context, 'options', {})
            
            # Create the plan
            plan = await self.create_plan(
                document_info=document_info,
                user_preferences=options,
                crew_type=self.crew_type
            )
            
            # Store plan in context
            context.agent_instructions = plan
            
            # Log success
            logger.info(f"PlannerAgent created plan with instructions for {len(plan)} agents")
            
            return plan
            
        except Exception as e:
            logger.error(f"Error in planning: {e}")
            logger.error(traceback.format_exc())
            
            # Create fallback plan in case of error
            fallback_plan = self._create_basic_plan(self._get_agent_types())
            context.agent_instructions = fallback_plan
            
            logger.info(f"Created fallback plan due to error")
            return fallback_plan
    
    async def create_plan(
        self, 
        document_info: Dict[str, Any],
        user_preferences: Dict[str, Any],
        crew_type: str
    ) -> Dict[str, Dict[str, str]]:
        """
        Create tailored instructions for each agent in a crew.
        
        Args:
            document_info: Document metadata
            user_preferences: User preferences
            crew_type: Type of crew
            
        Returns:
            Plan dictionary with instructions for each agent
        """
        with self.execution_tracking():
            # Get the available agent types for this crew
            agent_types = self._get_agent_types()
            
            # Create a planning context
            planning_context = {
                "document_info": document_info,
                "user_preferences": user_preferences,
                "crew_type": crew_type,
                "agent_types": agent_types
            }
            
            # Add issue definition if applicable
            if crew_type == "issues" and "issue_definition" in self.config:
                planning_context["issue_definition"] = self.config["issue_definition"]
            
            # Execute the planning task
            try:
                result = await self.execute_task(planning_context)
                
                # Ensure result is in the correct format
                plan = self._normalize_plan_format(result, agent_types)
                
                # Add metadata
                plan["_metadata"] = {
                    "created_at": datetime.now().isoformat(),
                    "crew_type": crew_type,
                    "detail_level": user_preferences.get("detail_level", "standard")
                }
                
                return plan
                
            except Exception as e:
                logger.error(f"Error in create_plan: {e}")
                logger.error(traceback.format_exc())
                
                # Create fallback plan in case of error
                return self._create_basic_plan(agent_types)
    
    def _get_agent_types(self) -> List[str]:
        """
        Get the list of agent types configured for this crew.
        
        Returns:
            List of agent type strings
        """
        # Get from workflow configuration
        stages = self.config.get("workflow", {}).get("enabled_stages", [])
        
        # Filter to just agent stages (not document_analysis or chunking)
        agent_stages = [stage for stage in stages if stage not in ["document_analysis", "chunking"]]
        
        # Return agent types or default set if not configured
        if agent_stages:
            return agent_stages
        
        # Default agent types
        return ["planner", "extractor", "aggregator", "evaluator", "formatter", "reviewer"]
    
    def _normalize_plan_format(self, plan_result: Any, agent_types: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Ensure the plan result is in the expected format with improved validation.
        
        Args:
            plan_result: Result from LLM planning
            agent_types: List of agent types to include
            
        Returns:
            Normalized plan dictionary
        """
        # Start with an empty plan
        plan_dict = {}
        
        # If result is a string, try to parse as JSON
        if isinstance(plan_result, str):
            try:
                parsed_result = self.parse_llm_json(plan_result, {})
                if isinstance(parsed_result, dict):
                    plan_dict = parsed_result
                else:
                    logger.warning(f"Parsed result is not a dictionary: {type(parsed_result)}")
            except Exception as e:
                logger.error(f"Error parsing plan result: {e}")
        elif isinstance(plan_result, dict):
            # Use result directly
            plan_dict = plan_result
        else:
            logger.warning(f"Unexpected plan result type: {type(plan_result)}")
        
        # Validate plan structure for all required agent types
        for agent_type in agent_types:
            # Skip if agent already has valid instructions
            if agent_type in plan_dict and isinstance(plan_dict[agent_type], dict):
                agent_plan = plan_dict[agent_type]
                
                # Ensure required fields exist
                if "instructions" not in agent_plan:
                    logger.warning(f"Missing instructions for {agent_type} in plan, adding default")
                    agent_plan["instructions"] = self._get_default_instructions(agent_type)
                
                if "emphasis" not in agent_plan:
                    agent_plan["emphasis"] = self._get_default_emphasis()
                    
            else:
                # Create default instructions for this agent
                logger.warning(f"Missing or invalid plan for {agent_type}, creating default")
                plan_dict[agent_type] = self._create_agent_instructions(agent_type)
        
        return plan_dict
    
    def _create_basic_plan(self, agent_types: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Create a basic plan with default instructions for each agent.
        Used as a fallback when planning fails.
        
        Args:
            agent_types: List of agent types to include
            
        Returns:
            Basic plan dictionary
        """
        logger.info(f"Creating basic plan for {len(agent_types)} agent types")
        plan = {}
        
        for agent_type in agent_types:
            plan[agent_type] = self._create_agent_instructions(agent_type)
        
        return plan
    
    def _create_agent_instructions(self, agent_type: str) -> Dict[str, str]:
        """
        Create instructions for a specific agent using config.
        
        Args:
            agent_type: Type of agent
            
        Returns:
            Instructions dictionary
        """
        # Get role information from config
        role_info = self.config.get("workflow", {}).get("agent_roles", {}).get(agent_type, {})
        
        return {
            "instructions": role_info.get("primary_task", self._get_default_instructions(agent_type)),
            "emphasis": self._get_default_emphasis()
        }
    
    def _get_default_instructions(self, agent_type: str) -> str:
        """
        Get default instructions for an agent type.
        
        Args:
            agent_type: Type of agent
            
        Returns:
            Default instruction string
        """
        # Default instructions by agent type
        defaults = {
            "extractor": f"Identify {self.crew_type} in each document chunk",
            "aggregator": f"Combine similar {self.crew_type} from all document chunks",
            "evaluator": f"Evaluate the severity and impact of each {self.crew_type}",
            "formatter": f"Create a clear, organized report of {self.crew_type}",
            "reviewer": "Review the analysis for quality and alignment with user needs"
        }
        
        return defaults.get(agent_type, f"Process {self.crew_type} data")
    
    def _get_default_emphasis(self) -> str:
        """
        Get default emphasis based on crew type.
        
        Returns:
            Default emphasis string
        """
        emphasis_map = {
            "issues": "Focus on identifying problems that impact objectives or efficiency",
            "actions": "Focus on clear, actionable items with ownership and deadlines",
            "insights": "Focus on extracting meaningful patterns and observations"
        }
        
        return emphasis_map.get(self.crew_type, "")
    
    def _get_stage_specific_content(self, context) -> str:
        """
        Get stage-specific content for the planning prompt.
        
        Args:
            context: Planning context
            
        Returns:
            Stage-specific content string
        """
        # Extract info from context
        if isinstance(context, dict):
            agent_types = context.get("agent_types", [])
            issue_definition = context.get("issue_definition", {})
            
            # Create agent roles description
            agent_roles = []
            for agent_type in agent_types:
                role_info = self.config.get("workflow", {}).get("agent_roles", {}).get(agent_type, {})
                
                # Skip planner itself
                if agent_type == "planner":
                    continue
                    
                description = role_info.get("description", f"{agent_type.capitalize()} Agent")
                primary_task = role_info.get("primary_task", f"Process {self.crew_type}")
                
                agent_roles.append(f"{agent_type.upper()}: {description}\n  Primary task: {primary_task}")
            
            # Create issue definition text if applicable
            issue_def_text = ""
            if self.crew_type == "issues" and issue_definition:
                issue_def_text = "ISSUE DEFINITION:\n"
                issue_def_text += issue_definition.get('description', 'Any problem, challenge, risk, or concern')
                issue_def_text += "\n\nSEVERITY LEVELS:\n"
                
                severity_levels = issue_definition.get("severity_levels", {})
                for level, desc in severity_levels.items():
                    issue_def_text += f"- {level}: {desc}\n"
            
            # Build the final content without using problematic indentation in f-strings
            content = "AGENT ROLES:\n"
            content += '\n\n'.join(agent_roles)
            
            if issue_def_text:
                content += f"\n\n{issue_def_text}"
            
            # Add user preferences if available
            user_prefs = context.get("user_preferences", {})
            if user_prefs:
                content += "\n\nUSER PREFERENCES:\n"
                
                # Add detail level
                detail_level = user_prefs.get("detail_level", "standard")
                content += f"Detail Level: {detail_level}\n"
                
                # Add focus areas
                focus_areas = user_prefs.get("focus_areas", [])
                if focus_areas:
                    content += f"Focus Areas: {', '.join(focus_areas)}\n"
                
                # Add custom instructions
                custom_instructions = user_prefs.get("user_instructions", "")
                if custom_instructions:
                    content += f"Custom Instructions: {custom_instructions}\n"
            
            # Add document info if available
            doc_info = context.get("document_info", {})
            if doc_info:
                content += "\n\nDOCUMENT CHARACTERISTICS:\n"
                
                # Add transcript info if available
                is_transcript = doc_info.get("is_meeting_transcript", False)
                content += f"Is Meeting Transcript: {is_transcript}\n"
                
                # Add key topics if available
                if "preview_analysis" in doc_info and "key_topics" in doc_info["preview_analysis"]:
                    topics = doc_info["preview_analysis"]["key_topics"]
                    if topics:
                        content += f"Key Topics: {', '.join(topics[:5])}\n"
            
            content += "\n\nPLANNING TASK:\n"
            content += "Create personalized instructions for each agent type based on the document and user preferences.\n"
            content += "For each agent, provide specific \"instructions\" and \"emphasis\" fields that are tailored to this particular document and analysis."
            
            return content
        
        return ""