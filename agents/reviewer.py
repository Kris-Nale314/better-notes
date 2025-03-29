"""
Streamlined Reviewer Agent for Better Notes that assesses output quality.
Provides a simple quality assessment of the formatted report.
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
    Streamlined Reviewer agent that assesses the quality of the analysis.
    Performs a final quality check and provides feedback.
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
        
        logger.info(f"ReviewerAgent initialized for {crew_type}")
    
    async def process(self, context):
        """
        Review the formatted report and provide quality assessment.
        
        Args:
            context: ProcessingContext object
            
        Returns:
            Review results
        """
        logger.info("ReviewerAgent starting review process")
        
        try:
            # Get formatted report from context
            formatted_result = context.results.get("formatting", "")
            
            if not formatted_result:
                logger.warning("No formatted report found for review")
                return self._create_simple_review("No formatted report available to review.")
            
            # Review the report
            review_result = await self.review_analysis(
                formatted_result=formatted_result,
                document_info=getattr(context, 'document_info', {}),
                user_preferences=getattr(context, 'options', {})
            )
            
            logger.info("Successfully completed review")
            return review_result
            
        except Exception as e:
            logger.error(f"Error in review process: {e}")
            
            # Return simple review in case of error
            return self._create_simple_review(f"Error during review: {str(e)}")
    
    async def review_analysis(
        self,
        formatted_result: Any,
        document_info: Optional[Dict[str, Any]] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Review a formatted result for quality and alignment with user expectations.
        
        Args:
            formatted_result: The formatted result to review
            document_info: Optional document metadata
            user_preferences: Optional user preferences
            
        Returns:
            Review results with assessment
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
        
        # Add focus area guidance if applicable
        focus_areas = user_preferences.get("focus_areas", []) if user_preferences else []
        if focus_areas:
            review_context["focus_area_guidance"] = self._get_focus_area_guidance(focus_areas)
        
        # Add detail level guidance if applicable
        detail_level = user_preferences.get("detail_level", "standard") if user_preferences else "standard"
        review_context["detail_guidance"] = self._get_detail_level_guidance(detail_level)
        
        # Execute review
        try:
            result = await self.execute_task(review_context)
            
            # Process and normalize the review result
            processed_result = self._process_review_result(result)
            
            return processed_result
            
        except Exception as e:
            logger.error(f"Error in review_analysis: {e}")
            
            # Return simple review in case of error
            return self._create_simple_review(f"Error during review: {str(e)}")
    
    def _get_stage_specific_content(self, context) -> str:
        """
        Get stage-specific content for the prompt.
        
        Args:
            context: Review context
            
        Returns:
            Stage-specific content string
        """
        if isinstance(context, dict):
            # Build the review prompt content
            content_parts = []
            
            # Add review criteria
            if "review_criteria" in context:
                criteria = context["review_criteria"]
                content_parts.append("REVIEW CRITERIA:")
                for key, value in criteria.items():
                    content_parts.append(f"- {key}: {value}")
            
            # Add user preferences
            if "user_preferences" in context and context["user_preferences"]:
                preferences = context["user_preferences"]
                
                content_parts.append("\nUSER PREFERENCES:")
                
                # Add detail level
                detail_level = preferences.get("detail_level", "standard")
                content_parts.append(f"Detail Level: {detail_level}")
                
                # Add focus areas
                focus_areas = preferences.get("focus_areas", [])
                if focus_areas:
                    content_parts.append(f"Focus Areas: {', '.join(focus_areas)}")
                
                # Add custom instructions
                custom_instructions = preferences.get("user_instructions", "")
                if custom_instructions:
                    content_parts.append(f"Custom Instructions: {custom_instructions}")
            
            # Add focus area guidance
            if "focus_area_guidance" in context:
                content_parts.append("\nFOCUS AREA GUIDANCE:")
                content_parts.append(context["focus_area_guidance"])
            
            # Add detail level guidance
            if "detail_guidance" in context:
                content_parts.append("\nDETAIL LEVEL GUIDANCE:")
                content_parts.append(context["detail_guidance"])
            
            # Add review instructions
            content_parts.append("\nREVIEW TASK:")
            content_parts.append("1. Assess the quality of the analysis based on the criteria above")
            content_parts.append("2. Check if the analysis aligns with user preferences")
            content_parts.append("3. Score each criterion on a scale of 1-5 (1=poor, 5=excellent)")
            content_parts.append("4. Provide a summary assessment of the overall quality")
            content_parts.append("5. Suggest 1-2 specific improvements if needed")
            
            # Add report to review (possibly truncated)
            if "formatted_result" in context:
                content_parts.append("\nRESULT TO REVIEW:")
                
                # Truncate if needed to prevent token overload
                formatted_result = context["formatted_result"]
                if len(formatted_result) > 3000:
                    content_parts.append(formatted_result[:3000] + "\n...(content truncated for brevity)...")
                else:
                    content_parts.append(formatted_result)
            
            return "\n\n".join(content_parts)
        
        return ""
    
    def _prepare_content_for_review(self, formatted_result: Any) -> str:
        """
        Prepare content for review by extracting text.
        
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
        # Try to get from config
        focus_area_info = self.config.get("user_options", {}).get("focus_areas", {})
        
        focus_descriptions = []
        for area in focus_areas:
            if area in focus_area_info:
                focus_descriptions.append(f"{area}: {focus_area_info[area]}")
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
                    focus_descriptions.append(f"{area}: {defaults[area]}")
                else:
                    focus_descriptions.append(f"{area}: Focus on {area.lower()}-related aspects")
        
        return "\n".join(focus_descriptions)
    
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
        # Parse the result if it's a string
        if isinstance(result, str):
            try:
                parsed_result = self.parse_llm_json(result)
                if isinstance(parsed_result, dict):
                    result = parsed_result
                else:
                    return self._create_simple_review(result)
            except Exception:
                return self._create_simple_review(result)
        
        # Ensure result is a dictionary
        if not isinstance(result, dict):
            return self._create_simple_review(str(result))
        
        # Create standard result structure
        review_result = {
            "summary": result.get("summary", "Review completed."),
            "meets_requirements": result.get("meets_requirements", True),
            "assessment": {}
        }
        
        # Extract assessment scores
        assessment = result.get("assessment", {})
        if isinstance(assessment, dict):
            review_result["assessment"] = assessment
        
        # Extract improvement suggestions
        suggestions = result.get("improvement_suggestions", [])
        if suggestions:
            review_result["improvement_suggestions"] = suggestions
        
        # Calculate overall score if not provided
        if "overall_score" not in review_result:
            scores = [v for k, v in assessment.items() if isinstance(v, (int, float))]
            if scores:
                review_result["overall_score"] = round(sum(scores) / len(scores), 1)
        
        # Add confidence
        if "confidence" not in review_result:
            overall_score = review_result.get("overall_score")
            if overall_score:
                if overall_score >= 4.0:
                    review_result["confidence"] = "high"
                elif overall_score >= 3.0:
                    review_result["confidence"] = "medium"
                else:
                    review_result["confidence"] = "low"
        
        return review_result
    
    def _create_simple_review(self, message: str) -> Dict[str, Any]:
        """
        Create a simple review result when normal review fails.
        
        Args:
            message: Review message or error
            
        Returns:
            Simple review result
        """
        return {
            "summary": message,
            "meets_requirements": True,
            "assessment": {
                "overall": 3
            },
            "confidence": "medium",
            "simplified": True,
            "timestamp": datetime.now().isoformat()
        }