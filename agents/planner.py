"""
Planner Agent - Creates optimized, document-aware instructions for specialized agents.
"""

from typing import Dict, Any, List, Optional
import json
import logging
from datetime import datetime
from .base import BaseAgent

logger = logging.getLogger(__name__)

class PlannerAgent(BaseAgent):
    """
    Meta-agent that creates tailored plans for document analysis crews.
    """
    
    def __init__(
        self,
        llm_client,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        max_chunk_size: int = 2000,
        max_rpm: int = 10
    ):
        """Initialize a planner agent."""
        super().__init__(
            llm_client=llm_client,
            agent_type="planner",
            crew_type="meta",
            config=config,
            verbose=verbose,
            max_chunk_size=max_chunk_size,
            max_rpm=max_rpm
        )
    
    def create_plan(
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
        start_time = datetime.now()
        
        # Extract essential document info to reduce token usage
        simplified_doc_info = self._simplify_document_info(document_info)
        
        # Extract user preference details
        detail_level = user_preferences.get("detail_level", "standard")
        focus_areas = user_preferences.get("focus_areas", [])
        user_instructions = user_preferences.get("user_instructions", "")
        
        # Get the agent role definitions from config based on crew type
        role_definitions = self._get_agent_role_definitions(crew_type)
        
        # Create prompt with configuration-aware instructions
        prompt = self._build_planning_prompt(
            crew_type, 
            simplified_doc_info, 
            detail_level, 
            focus_areas, 
            user_instructions, 
            role_definitions
        )
        
        try:
            # Get planning result from LLM
            logger.info(f"Creating plan for {crew_type} crew")
            planning_result = self.llm_client.generate_completion(prompt)
            
            # Parse the result
            plan = self._parse_planning_result(planning_result, crew_type, user_preferences)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Successfully created plan in {execution_time:.2f}s")
            
            return plan
                
        except Exception as e:
            logger.error(f"Error in plan creation: {str(e)}")
            return self._create_fallback_plan(crew_type, user_preferences)
    
    def _simplify_document_info(self, document_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract only the most relevant information from document_info."""
        insights = {
            "document_type": "transcript" if document_info.get("is_meeting_transcript", False) else "document",
            "length": document_info.get("original_text_length", 0)
        }
        
        # Get summary if available
        if "preview_analysis" in document_info and "summary" in document_info["preview_analysis"]:
            insights["summary"] = document_info["preview_analysis"]["summary"]
        
        # Get key topics if available
        if "preview_analysis" in document_info and "key_topics" in document_info["preview_analysis"]:
            insights["key_topics"] = document_info["preview_analysis"]["key_topics"][:5]  # Top 5 only
        
        # Get document stats
        if "basic_stats" in document_info:
            stats = document_info["basic_stats"]
            insights["word_count"] = stats.get("word_count", 0)
            insights["sentence_count"] = stats.get("sentence_count", 0)
        
        return insights
    
    def _get_agent_role_definitions(self, crew_type: str) -> str:
        """Get agent role definitions based on crew type from configuration."""
        # Get the agents section from config
        crew_config = self.config_manager.get_config(crew_type)
        agents_config = crew_config.get("agents", {})
        
        # Build role definitions string from config
        roles = []
        for agent_type, config in agents_config.items():
            role = config.get("role", f"{agent_type.title()} Agent")
            goal = config.get("goal", f"Process {crew_type}")
            
            roles.append(f"{agent_type.upper()} AGENT: {role}\n  Goal: {goal}")
        
        if not roles:
            # If no roles defined in config, use default roles
            return self._get_default_role_definitions(crew_type)
            
        return "\n\n".join(roles)
    
    def _get_default_role_definitions(self, crew_type: str) -> str:
        """Get default role definitions when not available in config."""
        default_roles = {
            "extraction": f"{crew_type.title()} Extractor - Identifies items in document chunks",
            "aggregation": f"{crew_type.title()} Aggregator - Combines and deduplicates items",
            "evaluation": f"{crew_type.title()} Evaluator - Assesses importance and impact",
            "formatting": "Report Formatter - Creates clear, structured reports",
            "reviewer": "Analysis Reviewer - Ensures quality and alignment with user needs"
        }
        
        roles = []
        for agent_type, role in default_roles.items():
            roles.append(f"{agent_type.upper()} AGENT: {role}")
        
        return "\n\n".join(roles)
    
    def _build_planning_prompt(
        self, 
        crew_type: str, 
        doc_insights: Dict[str, Any],
        detail_level: str,
        focus_areas: List[str],
        user_instructions: str,
        role_definitions: str
    ) -> str:
        """Build the planning prompt for the LLM."""
        return f"""
        You are an expert Planner who creates tailored instructions for AI agents.
        
        DOCUMENT INFORMATION:
        {json.dumps(doc_insights, indent=2)}
        
        USER PREFERENCES:
        - Detail Level: {detail_level}
        - Focus Areas: {', '.join(focus_areas) if focus_areas else 'None specified'}
        - User Instructions: {user_instructions}
        
        CREW TYPE: {crew_type}
        
        Your task is to create specialized instructions for each agent in the analysis crew.
        Each agent has a specific role in the analysis process:
        
        {role_definitions}
        
        For each agent, provide:
        - instructions: Clear guidance tailored to their specific role and this document
        - emphasis: Aspects they should particularly focus on for this document
        
        Your instructions should incorporate:
        - The document type and characteristics
        - The user's desired detail level ({detail_level})
        - The specified focus areas
        - Any specific user instructions
        
        Return your response in this JSON format:
        {{
          "extraction": {{ "instructions": "...", "emphasis": "..." }},
          "aggregation": {{ "instructions": "...", "emphasis": "..." }},
          "evaluation": {{ "instructions": "...", "emphasis": "..." }},
          "formatting": {{ "instructions": "...", "emphasis": "..." }},
          "reviewer": {{ "instructions": "...", "emphasis": "..." }}
        }}
        """
    
    def _parse_planning_result(
        self,
        planning_result: str,
        crew_type: str,
        user_preferences: Dict[str, Any]
    ) -> Dict[str, Dict[str, str]]:
        """Parse the planning result from the LLM."""
        try:
            # Extract JSON portion if needed
            if "{" in planning_result and "}" in planning_result:
                json_start = planning_result.find("{")
                json_end = planning_result.rfind("}") + 1
                json_str = planning_result[json_start:json_end]
                plan = json.loads(json_str)
            else:
                plan = json.loads(planning_result)
            
            # Validate all required agents are present
            required_agents = ["extraction", "aggregation", "evaluation", "formatting", "reviewer"]
            for agent in required_agents:
                if agent not in plan:
                    # Use fallback for missing agent
                    plan[agent] = self._create_fallback_instructions(agent, crew_type, user_preferences)
            
            # Add metadata to plan
            plan["_metadata"] = {
                "created_at": datetime.now().isoformat(),
                "crew_type": crew_type,
                "detail_level": user_preferences.get("detail_level", "standard")
            }
            
            return plan
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse planning result as JSON")
            return self._create_fallback_plan(crew_type, user_preferences)
    
    def _create_fallback_plan(
        self, 
        crew_type: str, 
        user_preferences: Dict[str, Any]
    ) -> Dict[str, Dict[str, str]]:
        """Create a fallback plan when LLM planning fails."""
        logger.warning(f"Using fallback plan for {crew_type} crew")
        
        # Create fallback instructions for each agent
        plan = {
            "extraction": self._create_fallback_instructions("extraction", crew_type, user_preferences),
            "aggregation": self._create_fallback_instructions("aggregation", crew_type, user_preferences),
            "evaluation": self._create_fallback_instructions("evaluation", crew_type, user_preferences),
            "formatting": self._create_fallback_instructions("formatting", crew_type, user_preferences),
            "reviewer": self._create_fallback_instructions("reviewer", crew_type, user_preferences)
        }
        
        # Add metadata
        plan["_metadata"] = {
            "created_at": datetime.now().isoformat(),
            "crew_type": crew_type,
            "is_fallback": True
        }
        
        return plan
    
    def _create_fallback_instructions(
        self, 
        agent_type: str, 
        crew_type: str, 
        user_preferences: Dict[str, Any]
    ) -> Dict[str, str]:
        """Create fallback instructions for a specific agent."""
        # Extract preferences
        detail_level = user_preferences.get("detail_level", "standard")
        focus_areas = user_preferences.get("focus_areas", [])
        user_instructions = user_preferences.get("user_instructions", "")
        
        # Build base instruction
        base_instruction = f"Analyze with {detail_level} level of detail."
        if focus_areas:
            base_instruction += f" Focus on these areas: {', '.join(focus_areas)}."
        if user_instructions:
            base_instruction += f" User instructions: {user_instructions}"
        
        # Agent-specific instructions by crew type
        instructions_map = {
            "extraction": {
                "issues": f"Identify potential issues in this document chunk. {base_instruction}",
                "actions": f"Extract action items from this document chunk. {base_instruction}",
                "default": f"Extract relevant information from this document chunk. {base_instruction}"
            },
            "aggregation": {
                "issues": f"Combine similar issues while preserving important distinctions. {base_instruction}",
                "actions": f"Group related action items and remove duplicates. {base_instruction}",
                "default": f"Consolidate similar items from all chunks. {base_instruction}"
            },
            "evaluation": {
                "issues": f"Assess the severity and impact of each issue. {base_instruction}",
                "actions": f"Prioritize action items by importance. {base_instruction}",
                "default": f"Evaluate the importance of each item. {base_instruction}"
            },
            "formatting": {
                "issues": f"Create a structured report grouped by severity. {base_instruction}",
                "actions": f"Create an organized action items report. {base_instruction}",
                "default": f"Format findings into a clear report. {base_instruction}"
            },
            "reviewer": {
                "issues": f"Review for quality, accuracy and alignment with user needs. {base_instruction}",
                "actions": f"Verify completeness and clarity of action items. {base_instruction}",
                "default": f"Check quality and completeness of the report. {base_instruction}"
            }
        }
        
        # Get appropriate instruction
        agent_map = instructions_map.get(agent_type, {})
        instructions = agent_map.get(crew_type, agent_map.get("default", f"Process the {crew_type} data. {base_instruction}"))
        
        # Create emphasis based on detail level
        emphasis_map = {
            "essential": "Focus only on the most important elements.",
            "standard": "Balance detail and conciseness.",
            "comprehensive": "Provide thorough analysis with comprehensive detail."
        }
        emphasis = emphasis_map.get(detail_level, "Balance detail and conciseness.")
        
        # Add focus areas to emphasis
        if focus_areas:
            emphasis += f" Pay special attention to: {', '.join(focus_areas)}."
        
        return {
            "instructions": instructions,
            "emphasis": emphasis
        }