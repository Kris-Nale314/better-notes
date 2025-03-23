"""
Reviewer Agent - Ensures the analysis meets quality standards and user expectations.
Supports the metadata-layered approach through configuration.
"""

from typing import Dict, Any, Optional, List, Union, Tuple
import json
from datetime import datetime
from .base import BaseAgent

class ReviewerAgent(BaseAgent):
    """
    Agent specialized in reviewing analysis output to ensure quality and alignment.
    Acts as a final quality check before delivering results to the user.
    """
    
    def __init__(
        self,
        llm_client,
        crew_type: str,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        max_chunk_size: int = 1500,
        max_rpm: int = 10,  
        **kwargs # Accept any additional kwargs
    ):
        """
        Initialize a reviewer agent.
        
        Args:
            llm_client: LLM client for agent communication
            crew_type: Type of crew (issues, actions, opportunities)
            config: Optional pre-loaded configuration
            verbose: Whether to enable verbose mode
            max_chunk_size: Maximum size of text chunks to process
            max_rpm: Maximum requests per minute for API rate limiting
            custom_instructions: Custom instructions from Instructor agent
        """
        super().__init__(
            llm_client=llm_client,
            agent_type="reviewer",
            crew_type=crew_type,
            config=config,
            verbose=verbose,
            max_chunk_size=max_chunk_size,
            max_rpm=max_rpm,
            **kwargs # Pass any additional kwargs to the base class
        )
    
    def review_analysis(
        self,
        formatted_result: Any,
        document_info: Optional[Dict[str, Any]] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Review a formatted analysis result for quality and alignment with user expectations.
        
        Args:
            formatted_result: The formatted analysis to review
            document_info: Optional document metadata
            user_preferences: Optional user preferences
            
        Returns:
            Review results with assessment and suggestions
        """
        # Process the formatted result based on its type
        if isinstance(formatted_result, str):
            content_to_review = self.truncate_text(formatted_result, self.max_chunk_size)
        elif isinstance(formatted_result, dict):
            content_to_review = json.dumps(formatted_result, indent=2)[:self.max_chunk_size]
        else:
            content_to_review = str(formatted_result)[:self.max_chunk_size]
        
        # Extract key user preferences
        user_instructions = ""
        detail_level = "standard"
        focus_areas = []
        
        if user_preferences:
            user_instructions = user_preferences.get("user_instructions", "")
            detail_level = user_preferences.get("detail_level", "standard")
            focus_areas = user_preferences.get("focus_areas", [])
        
        # Get review criteria from config
        review_criteria = self.get_review_criteria()
        
        # Prepare context for review
        context = {
            "formatted_result": content_to_review,
            "document_info": document_info or {},
            "user_instructions": user_instructions,
            "detail_level": detail_level,
            "focus_areas": focus_areas,
            "review_criteria": review_criteria
        }
        
        # Get focus area guidance if applicable
        if focus_areas:
            focus_guidance = self._get_focus_area_guidance(focus_areas)
            if focus_guidance:
                context["focus_area_guidance"] = focus_guidance
        
        # Get detail level guidance
        detail_guidance = self._get_detail_level_guidance(detail_level)
        if detail_guidance:
            context["detail_level_guidance"] = detail_guidance
        
        # Execute the review task
        result = self.execute_task(context=context)
        
        # Enhance the result with review metadata
        result = self._enhance_result_with_metadata(result, user_preferences)
        
        return result
    
    def get_review_criteria(self) -> Dict[str, str]:
        """
        Get the review criteria for this crew type.
        
        Returns:
            Dictionary of review criteria
        """
        # Try to get from new structure first
        agent_config = self._get_agent_config()
        if "review_criteria" in agent_config:
            return agent_config["review_criteria"]
        
        # Default criteria
        return {
            "alignment": "Does the analysis align with user instructions and focus areas?",
            "completeness": "Does the report address all significant items at the appropriate detail level?",
            "consistency": "Are ratings applied consistently throughout the analysis?",
            "clarity": "Is the report clear, well-organized, and actionable?",
            "balance": "Are items presented in a balanced way without over or under-emphasis?"
        }
    
    def _get_focus_area_guidance(self, focus_areas: List[str]) -> str:
        """
        Get guidance for specific focus areas.
        
        Args:
            focus_areas: List of focus areas
            
        Returns:
            Guidance string for focus areas
        """
        # Get focus area info from config
        focus_area_info = self.config.get("user_options", {}).get("focus_areas", {})
        
        # Collect guidance for each selected focus area
        guidance_parts = []
        
        for area in focus_areas:
            if area in focus_area_info:
                area_config = focus_area_info[area]
                if "description" in area_config:
                    guidance_parts.append(f"{area}: {area_config['description']}")
                
                # Add review emphasis if available
                if "review_emphasis" in area_config:
                    guidance_parts.append(f"  Review emphasis: {area_config['review_emphasis']}")
        
        return "\n".join(guidance_parts)
    
    def _get_detail_level_guidance(self, detail_level: str) -> str:
        """
        Get guidance for the detail level.
        
        Args:
            detail_level: Detail level (essential, standard, comprehensive)
            
        Returns:
            Guidance string for the detail level
        """
        # Get detail level info from config
        detail_level_info = self.config.get("user_options", {}).get("detail_levels", {}).get(detail_level, {})
        
        if "description" in detail_level_info:
            # Get reviewer-specific guidance if available
            if "agent_guidance" in detail_level_info and "reviewer" in detail_level_info["agent_guidance"]:
                return f"{detail_level_info['description']}\nReviewer guidance: {detail_level_info['agent_guidance']['reviewer']}"
            else:
                return detail_level_info["description"]
        
        # Default descriptions
        default_guidance = {
            "essential": "Focus only on the most important elements with minimal detail.",
            "standard": "Provide a balanced amount of detail, covering all significant aspects.",
            "comprehensive": "Include thorough details, context, and explanations for all elements."
        }
        
        return default_guidance.get(detail_level, "")
    
    def _enhance_result_with_metadata(self, result: Any, user_preferences: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enhance the review result with additional metadata.
        
        Args:
            result: Review result
            user_preferences: User preferences
            
        Returns:
            Enhanced result with metadata
        """
        # Handle string results
        if isinstance(result, str):
            try:
                # Try to parse as JSON
                if result.strip().startswith('{') and result.strip().endswith('}'):
                    parsed_result = json.loads(result)
                    if isinstance(parsed_result, dict):
                        result = parsed_result
            except json.JSONDecodeError:
                # Not JSON, create a basic structure
                result = {
                    "review_result": {
                        "meets_requirements": None,
                        "summary": result,
                        "assessment": {}
                    }
                }
        
        # Handle non-dictionary results
        if not isinstance(result, dict):
            result = {
                "review_result": {
                    "meets_requirements": None,
                    "summary": str(result),
                    "assessment": {}
                }
            }
        
        # Ensure review_result exists
        if "review_result" not in result:
            result = {"review_result": result}
        
        # Ensure required fields exist
        review_result = result["review_result"]
        if "meets_requirements" not in review_result:
            review_result["meets_requirements"] = None
        
        if "assessment" not in review_result:
            review_result["assessment"] = {}
        
        if "summary" not in review_result:
            review_result["summary"] = "Review completed."
        
        # Calculate confidence if missing
        if "confidence" not in review_result:
            assessment = review_result["assessment"]
            if assessment and isinstance(assessment, dict):
                # Calculate confidence based on assessment scores
                scores = [v for k, v in assessment.items() if isinstance(v, (int, float))]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    if avg_score >= 4.0:
                        confidence = "high"
                    elif avg_score >= 3.0:
                        confidence = "medium"
                    else:
                        confidence = "low"
                    review_result["confidence"] = confidence
        
        # Add metadata
        result["_metadata"] = {
            "reviewed_at": datetime.now().isoformat(),
            "crew_type": self.crew_type,
            "detail_level": user_preferences.get("detail_level", "standard") if user_preferences else "standard",
            "focus_areas": user_preferences.get("focus_areas", []) if user_preferences else []
        }
        
        return result