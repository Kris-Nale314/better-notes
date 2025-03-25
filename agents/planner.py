"""
Planner Agent - Creates optimized, document-aware instructions for specialized agents.
Simplified implementation that leverages the new BaseAgent architecture.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base import BaseAgent

logger = logging.getLogger(__name__)

class PlannerAgent(BaseAgent):
    """
    Agent that creates tailored plans for document analysis.
    Simplified implementation with cleaner configuration access and improved error handling.
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
    
    async def process(self, context):
        """
        Create a plan using the provided context.
        
        Args:
            context: ProcessingContext object
            
        Returns:
            Planning result with instructions for each agent
        """
        # Execute planning
        plan = await self.create_plan(
            document_info=context.document_info,
            user_preferences=context.options,
            crew_type=self.crew_type
        )
        
        # Store plan in context
        context.agent_instructions = plan
        
        return plan
    
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
                "agent_types": agent_types,
                "issue_definition": self.config.get("issue_definition", {})
            }
            
            # Execute the planning task
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
        Ensure the plan result is in the expected format.
        
        Args:
            plan_result: Result from LLM planning
            agent_types: List of agent types to include
            
        Returns:
            Normalized plan dictionary
        """
        # If result is a string, try to parse as JSON
        if isinstance(plan_result, str):
            try:
                plan_dict = self.parse_llm_json(plan_result, {})
            except:
                # If JSON parsing fails, create a simple instruction-only plan
                return self._create_basic_plan(agent_types)
        else:
            # Use result directly
            plan_dict = plan_result if isinstance(plan_result, dict) else {}
        
        # Check if we have a valid plan for each agent type
        for agent_type in agent_types:
            if agent_type not in plan_dict or not isinstance(plan_dict[agent_type], dict):
                # Create basic instruction for this agent
                plan_dict[agent_type] = self._create_agent_instructions(agent_type)
            else:
                # Ensure required fields exist
                agent_plan = plan_dict[agent_type]
                if "instructions" not in agent_plan:
                    agent_plan["instructions"] = self._get_default_instructions(agent_type)
                if "emphasis" not in agent_plan:
                    agent_plan["emphasis"] = ""
        
        return plan_dict
    
    def _create_basic_plan(self, agent_types: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Create a basic plan with default instructions for each agent.
        
        Args:
            agent_types: List of agent types to include
            
        Returns:
            Basic plan dictionary
        """
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
            
            content += "\n\nPLANNING TASK:\n"
            content += "Create instructions for each agent type based on the document and user preferences.\n"
            content += "For each agent, provide specific \"instructions\" and \"emphasis\" fields."
            
            return content
        
        return ""