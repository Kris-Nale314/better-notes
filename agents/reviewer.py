"""
Reviewer Agent - Ensures the analysis meets quality standards and user expectations.
Clean implementation that leverages the new BaseAgent architecture.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base import BaseAgent

logger = logging.getLogger(__name__)

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
        config_manager = None,
        verbose: bool = True,
        max_chunk_size: int = 1500,
        max_rpm: int = 10
    ):
        """Initialize a reviewer agent."""
        super().__init__(
            llm_client=llm_client,
            agent_type="reviewer",
            crew_type=crew_type,
            config=config,
            config_manager=config_manager,
            verbose=verbose,
            max_chunk_size=max_chunk_size,
            max_rpm=max_rpm
        )
    
    async def process(self, context):
        """
        Process formatted report using the context.
        
        Args:
            context: ProcessingContext object
            
        Returns:
            Review results
        """
        # Get formatted report from context
        formatted_result = context.results.get("formatting", {})
        
        # Review the report
        review_result = await self.review_analysis(
            formatted_result=formatted_result,
            document_info=context.document_info,
            user_preferences=context.options
        )
        
        return review_result
    
    async def review_analysis(
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
        # Get review criteria
        review_criteria = self._get_review_criteria()
        
        # Prepare review context
        review_context = {
            "formatted_result": self._prepare_content_for_review(formatted_result),
            "document_info": document_info or {},
            "user_preferences": user_preferences or {},
            "review_criteria": review_criteria
        }
    
    def _get_focus_area_guidance(self, focus_areas: List[str]) -> str:
        """
        Get guidance for specific focus areas.
        
        Args:
            focus_areas: List of focus areas
            
        Returns:
            Guidance string for focus areas
        """
        # Try to get from config
        focus_area_info = self.config.get("user_options", {}).get("focus_areas", {})
        
        guidance_parts = []
        for area in focus_areas:
            if area in focus_area_info:
                guidance_parts.append(f"{area}: {focus_area_info[area]}")
            else:
                # Default descriptions
                defaults = {
                    "technical": "Technical issues related to implementation, architecture, or technology",
                    "process": "Process-related issues in workflows, procedures, or methodologies",
                    "resource": "Resource constraints with staffing, budget, time, or materials",
                    "quality": "Quality concerns regarding standards, testing, or performance",
                    "risk": "Risk-related issues including compliance, security, or strategic risks"
                }
                if area in defaults:
                    guidance_parts.append(f"{area}: {defaults[area]}")
                else:
                    guidance_parts.append(f"{area}: Focus on {area.lower()}-related aspects")
        
        return "\n".join(guidance_parts)
    
    def _get_detail_level_guidance(self, detail_level: str) -> str:
        """
        Get guidance for a specific detail level.
        
        Args:
            detail_level: Detail level (essential, standard, comprehensive)
            
        Returns:
            Guidance string
        """
        # Try to get from config
        detail_levels = self.config.get("user_options", {}).get("detail_levels", {})
        
        if detail_level in detail_levels:
            return detail_levels[detail_level]
        
        # Default guidance
        defaults = {
            "essential": "Focus only on the most important elements with minimal detail.",
            "standard": "Provide a balanced amount of detail, covering all significant aspects.",
            "comprehensive": "Include thorough details, context, and explanations for all elements."
        }
        
        return defaults.get(detail_level, defaults["standard"])
    
    def _process_review_result(self, result: Any) -> Dict[str, Any]:
        """
        Process and normalize review result.
        
        Args:
            result: Review result from LLM
            
        Returns:
            Normalized review result dictionary
        """
        # Ensure result is a dictionary
        if isinstance(result, str):
            try:
                # Try to parse as JSON
                parsed_result = self.parse_llm_json(result)
                if isinstance(parsed_result, dict):
                    result = parsed_result
                else:
                    # Create basic structure
                    result = {
                        "review_result": {
                            "meets_requirements": None,
                            "summary": result,
                            "assessment": {}
                        }
                    }
            except:
                # Not valid JSON, create basic structure
                result = {
                    "review_result": {
                        "meets_requirements": None,
                        "summary": result,
                        "assessment": {}
                    }
                }
        elif not isinstance(result, dict):
            # Wrap non-dictionary result
            result = {
                "review_result": {
                    "meets_requirements": None,
                    "summary": str(result),
                    "assessment": {}
                }
            }
        
        # Check if review_result key exists
        if "review_result" not in result:
            # Assume the entire result is the review_result
            result = {"review_result": result}
        
        # Ensure required fields exist in review_result
        review_result = result["review_result"]
        if not isinstance(review_result, dict):
            # Convert to dictionary
            review_result = {
                "meets_requirements": None,
                "summary": str(review_result),
                "assessment": {}
            }
            result["review_result"] = review_result
        
        # Add required fields if missing
        if "meets_requirements" not in review_result:
            review_result["meets_requirements"] = None
        
        if "summary" not in review_result:
            review_result["summary"] = "Review completed."
        
        if "assessment" not in review_result or not isinstance(review_result["assessment"], dict):
            review_result["assessment"] = {}
        
        # Calculate confidence if missing
        if "confidence" not in review_result:
            assessment = review_result["assessment"]
            if assessment:
                # Calculate based on assessment scores
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
                else:
                    review_result["confidence"] = "medium"
            else:
                review_result["confidence"] = "medium"
        
        # Add metadata
        result["_metadata"] = {
            "reviewed_at": datetime.now().isoformat(),
            "crew_type": self.crew_type
        }
        
        return result
    
    def _get_stage_specific_content(self, context) -> str:
        """Get stage-specific content for the prompt."""
        if isinstance(context, dict):
            content_parts = []
            
            # Add review criteria
            if "review_criteria" in context:
                criteria = context["review_criteria"]
                criteria_str = "\n".join([f"- {key}: {value}" for key, value in criteria.items()])
                content_parts.append(f"REVIEW CRITERIA:\n{criteria_str}")
            
            # Add user preferences
            if "user_preferences" in context and context["user_preferences"]:
                preferences = context["user_preferences"]
                # Extract key preferences
                key_prefs = {
                    "detail_level": preferences.get("detail_level", "standard"),
                    "focus_areas": preferences.get("focus_areas", []),
                    "user_instructions": preferences.get("user_instructions", "")
                }
                content_parts.append(f"USER PREFERENCES:\n{json.dumps(key_prefs, indent=2)}")
            
            # Add focus area guidance
            if "focus_area_guidance" in context:
                content_parts.append(f"FOCUS AREA GUIDANCE:\n{context['focus_area_guidance']}")
            
            # Add detail level guidance
            if "detail_guidance" in context:
                content_parts.append(f"DETAIL LEVEL GUIDANCE:\n{context['detail_guidance']}")
            
            # Add content to review (possibly truncated)
            if "formatted_result" in context:
                formatted_result = context["formatted_result"]
                # Truncate if needed
                if len(formatted_result) > 3000:
                    formatted_result = formatted_result[:3000] + "\n...(content truncated for brevity)..."
                content_parts.append(f"CONTENT TO REVIEW:\n{formatted_result}")
            
            return "\n\n".join(content_parts)
        
        return ""
    
    def _prepare_content_for_review(self, formatted_result: Any) -> str:
        """
        Prepare content for review by extracting text or converting to string.
        
        Args:
            formatted_result: Formatted result (could be string, dict, etc.)
            
        Returns:
            String representation suitable for review
        """
        # Handle HTML string
        if isinstance(formatted_result, str) and self._is_html_content(formatted_result):
            # Strip HTML tags for easier review
            return self._strip_html_tags(formatted_result)
        
        # Handle dictionary with formatted_report key
        if isinstance(formatted_result, dict) and "formatted_report" in formatted_result:
            report = formatted_result["formatted_report"]
            if isinstance(report, str) and self._is_html_content(report):
                return self._strip_html_tags(report)
            return str(report)
        
        # Default to string representation
        return str(formatted_result)
    
    def _is_html_content(self, content: str) -> bool:
        """
        Determine if content is HTML.
        
        Args:
            content: Content to check
            
        Returns:
            Boolean indicating if content is HTML
        """
        if not isinstance(content, str):
            return False
            
        html_indicators = ["<html", "<div", "<h1", "<p>", "<table", "<body", "<head"]
        return any(indicator in content.lower() for indicator in html_indicators)
    
    def _strip_html_tags(self, html: str) -> str:
        """
        Strip HTML tags from content for easier review.
        
        Args:
            html: HTML content
            
        Returns:
            Plain text content
        """
        # Simple regex to remove tags
        text = re.sub(r'<[^>]+>', ' ', html)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Restore some structure
        text = text.replace(' h1 ', '\n\n# ')
        text = text.replace(' h2 ', '\n\n## ')
        text = text.replace(' h3 ', '\n\n### ')
        text = text.replace(' p ', '\n\n')
        
        return text.strip()
    
    def _get_review_criteria(self) -> Dict[str, str]:
        """
        Get review criteria based on crew type.
        
        Returns:
            Dictionary of review criteria
        """
        # Try to get from config
        agent_roles = self.config.get("workflow", {}).get("agent_roles", {})
        if "reviewer" in agent_roles and "review_criteria" in agent_roles["reviewer"]:
            return agent_roles["reviewer"]["review_criteria"]
        
        # Default criteria
        return {
            "alignment": "Does the analysis align with user instructions and focus areas?",
            "completeness": "Does the report address all significant items at the appropriate detail level?",
            "consistency": "Are ratings/evaluations applied consistently throughout the analysis?",
            "clarity": "Is the report clear, well-organized, and actionable?",
            "balance": "Are items presented in a balanced way without over or under-emphasis?"
        }