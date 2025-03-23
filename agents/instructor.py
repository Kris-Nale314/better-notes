"""
Instructor Agent - Creates tailored instructions for each agent in a crew.
Adapts to document context and user preferences to optimize agent performance.
Supports the metadata-layered approach through configuration.
"""

from typing import Dict, Any, Optional, List, Tuple
import json
import logging
from datetime import datetime
from .base import BaseAgent

logger = logging.getLogger(__name__)

class InstructorAgent(BaseAgent):
    """
    Meta-agent that creates tailored instructions for each agent in a crew.
    Adapts to document context and user preferences to optimize agent performance.
    """
    
    def __init__(
        self,
        llm_client,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        max_chunk_size: int = 1500,
        max_rpm: int = 10
    ):
        """
        Initialize an instructor agent.
        
        Args:
            llm_client: LLM client for agent communication
            config: Optional pre-loaded configuration
            verbose: Whether to enable verbose mode
            max_chunk_size: Maximum size of text chunks
            max_rpm: Maximum requests per minute
        """
        super().__init__(
            llm_client=llm_client,
            agent_type="instructor",
            crew_type="meta",
            config=config,
            verbose=verbose,
            max_chunk_size=max_chunk_size,
            max_rpm=max_rpm
        )
    
    def create_instructions(
        self, 
        document_info: Dict[str, Any],
        user_preferences: Dict[str, Any],
        crew_type: str
    ) -> Dict[str, Dict[str, str]]:
        """
        Create tailored instructions for each agent in a crew.
        
        Args:
            document_info: Document metadata and analysis
            user_preferences: User preferences for analysis
            crew_type: Type of analysis crew (issues, actions, etc.)
            
        Returns:
            Dictionary with tailored instructions for each agent
        """
        # Track start time for performance monitoring
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
        prompt = f"""
        You are an expert Instruction Architect who creates tailored instructions for AI agents.
        
        DOCUMENT INFORMATION:
        {json.dumps(simplified_doc_info, indent=2)}
        
        USER PREFERENCES:
        - Detail Level: {detail_level}
        - Focus Areas: {', '.join(focus_areas) if focus_areas else 'None specified'}
        - User Instructions: {user_instructions}
        
        CREW TYPE: {crew_type}
        
        Your task is to create specialized instructions for each agent in the analysis crew.
        Each agent has a specific role in the analysis process:
        
        {role_definitions}
        
        For each agent, provide:
        - instructions: Clear guidance tailored to their specific role
        - emphasis: Aspects they should particularly focus on
        
        Your instructions should incorporate:
        - The document type and characteristics
        - The user's desired detail level ({detail_level})
        - The specified focus areas: {', '.join(focus_areas) if focus_areas else 'None'}
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
        
        # Try multiple methods to get the result
        try:
            # Method 1: Use direct LLM call
            logger.info(f"Using direct LLM call for instruction creation ({crew_type})")
            result = self.llm_client.generate_completion(prompt)
            
            # Parse the result
            try:
                # Extract JSON from the result if needed
                if "{" in result and "}" in result:
                    json_start = result.find("{")
                    json_end = result.rfind("}") + 1
                    json_str = result[json_start:json_end]
                    parsed_result = json.loads(json_str)
                else:
                    parsed_result = json.loads(result)
                
                # Validate the result has all necessary agents
                required_agents = ["extraction", "aggregation", "evaluation", "formatting", "reviewer"]
                for agent in required_agents:
                    if agent not in parsed_result:
                        # Fill missing agent with fallback
                        parsed_result[agent] = self._create_agent_fallback(agent, detail_level, focus_areas, user_instructions, crew_type)
                
                # Calculate execution time
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Successfully created agent instructions in {execution_time:.2f}s")
                
                return parsed_result
                
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse instruction result as JSON: {e}")
                
                # Try to extract JSON with regex as a fallback
                import re
                json_match = re.search(r'({[\s\S]*})', result)
                if json_match:
                    try:
                        parsed_result = json.loads(json_match.group(1))
                        return parsed_result
                    except:
                        pass
        
        except Exception as e:
            logger.error(f"Error in instruction creation: {str(e)}")
        
        # If all else fails, use fallback instructions
        logger.warning("Using fallback instructions")
        return self._create_fallback_instructions(detail_level, focus_areas, user_instructions, crew_type)
    
    def _simplify_document_info(self, document_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simplify document info to reduce token usage.
        
        Args:
            document_info: Full document info
            
        Returns:
            Simplified document info
        """
        simplified = {}
        
        # Include only essential information
        is_transcript = document_info.get("is_meeting_transcript", False)
        simplified["document_type"] = "transcript" if is_transcript else "document"
        
        # Add document length if available
        if "original_text_length" in document_info:
            simplified["document_length"] = document_info["original_text_length"]
        
        # Add key topics if available
        if "preview_analysis" in document_info and isinstance(document_info["preview_analysis"], dict):
            preview = document_info["preview_analysis"]
            if "summary" in preview:
                simplified["summary"] = preview["summary"]
            if "key_topics" in preview:
                simplified["key_topics"] = preview["key_topics"][:5]  # Only include top 5
        
        # Add basic stats if available
        if "basic_stats" in document_info:
            simplified["stats"] = {
                "word_count": document_info["basic_stats"].get("word_count", 0),
                "paragraph_count": document_info["basic_stats"].get("paragraph_count", 0),
                "sentence_count": document_info["basic_stats"].get("sentence_count", 0)
            }
        
        return simplified
    
    def _get_agent_role_definitions(self, crew_type: str) -> str:
        """
        Get agent role definitions based on crew type from configuration.
        
        Args:
            crew_type: Type of crew
            
        Returns:
            String with agent role definitions
        """
        # Get the agents section from config
        agents_config = self.config.get("agents", {})
        
        # Build role definitions string from config
        roles = []
        for agent_type, config in agents_config.items():
            role = config.get("role", f"{agent_type.title()} Agent")
            goal = config.get("goal", f"Process {crew_type}")
            
            roles.append(f"{agent_type.upper()} AGENT: {role}\n  Goal: {goal}")
        
        return "\n\n".join(roles)
    
    def _create_agent_fallback(self, agent_type: str, detail_level: str, focus_areas: List[str], user_instructions: str, crew_type: str) -> Dict[str, str]:
        """
        Create a fallback instruction for a specific agent.
        
        Args:
            agent_type: Type of agent
            detail_level: Detail level
            focus_areas: Focus areas
            user_instructions: User instructions
            crew_type: Type of crew
            
        Returns:
            Dictionary with instructions and emphasis
        """
        # Create a base instruction that includes user preferences
        base_instruction = f"Analyze with {detail_level} level of detail."
        if focus_areas:
            base_instruction += f" Focus on these areas: {', '.join(focus_areas)}."
        if user_instructions:
            base_instruction += f" User instructions: {user_instructions}"
        
        # Get default instructions from config if available
        agent_config = self.config.get("agents", {}).get(agent_type, {})
        instructions = agent_config.get("instructions", f"Process {crew_type} through {agent_type}.")
        
        # Add base instruction to default instructions
        instructions = f"{instructions} {base_instruction}"
        
        # Create emphasis based on detail level
        if detail_level == "essential":
            emphasis = "Focus only on the most important items."
        elif detail_level == "comprehensive":
            emphasis = "Provide thorough analysis with all relevant details."
        else:  # standard
            emphasis = "Maintain a balanced approach between detail and conciseness."
        
        # Add focus areas to emphasis
        if focus_areas:
            emphasis += f" Pay special attention to: {', '.join(focus_areas)}."
        
        return {
            "instructions": instructions,
            "emphasis": emphasis
        }
    
    def _create_fallback_instructions(
        self,
        detail_level: str,
        focus_areas: List[str],
        user_instructions: str,
        crew_type: str
    ) -> Dict[str, Dict[str, str]]:
        """
        Create basic fallback instructions that still incorporate user preferences.
        
        Args:
            detail_level: Detail level setting
            focus_areas: Focus areas list
            user_instructions: User's custom instructions
            crew_type: Type of crew
            
        Returns:
            Dictionary with basic instructions for each agent
        """
        # Create fallback instructions for each agent
        return {
            "extraction": self._create_agent_fallback("extraction", detail_level, focus_areas, user_instructions, crew_type),
            "aggregation": self._create_agent_fallback("aggregation", detail_level, focus_areas, user_instructions, crew_type),
            "evaluation": self._create_agent_fallback("evaluation", detail_level, focus_areas, user_instructions, crew_type),
            "formatting": self._create_agent_fallback("formatting", detail_level, focus_areas, user_instructions, crew_type),
            "reviewer": self._create_agent_fallback("reviewer", detail_level, focus_areas, user_instructions, crew_type)
        }