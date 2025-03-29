"""
Streamlined Evaluator Agent for Better Notes that assesses item importance and impact.
Analyzes, categorizes, and prioritizes aggregated items.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base import BaseAgent

logger = logging.getLogger(__name__)

class EvaluatorAgent(BaseAgent):
    """
    Streamlined Evaluator agent that assesses aggregated items for severity and impact.
    Adds assessment metadata to prepare items for formatting.
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
        """Initialize an evaluator agent."""
        super().__init__(
            llm_client=llm_client,
            agent_type="evaluator",
            crew_type=crew_type,
            config=config,
            config_manager=config_manager,
            verbose=verbose,
            max_chunk_size=max_chunk_size,
            max_rpm=max_rpm
        )
        
        logger.info(f"EvaluatorAgent initialized for {crew_type}")
    
    async def process(self, context):
        """
        Process aggregated results using the context.
        
        Args:
            context: ProcessingContext object
            
        Returns:
            Evaluated results
        """
        logger.info("EvaluatorAgent starting evaluation process")
        
        try:
            # Get aggregation results from context
            aggregated_result = context.results.get("aggregation", {})
            
            if not aggregated_result:
                logger.warning("No aggregation results found for evaluation")
                return self._create_empty_result()
            
            # Evaluate results
            evaluated_result = await self.evaluate_items(
                aggregated_items=aggregated_result,
                document_info=getattr(context, 'document_info', {}),
                user_preferences=getattr(context, 'options', {})
            )
            
            logger.info("Successfully evaluated aggregated items")
            return evaluated_result
            
        except Exception as e:
            logger.error(f"Error in evaluation process: {e}")
            
            # Return empty result in case of error
            return self._create_empty_result()
    
    async def evaluate_items(
        self, 
        aggregated_items: Dict[str, Any], 
        document_info: Optional[Dict[str, Any]] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate aggregated items for severity, importance, or impact.
        
        Args:
            aggregated_items: Items to evaluate
            document_info: Optional document metadata
            user_preferences: Optional user preferences
            
        Returns:
            Evaluated items with ratings and rationales
        """
        # Get evaluation criteria
        evaluation_criteria = self._get_evaluation_criteria()
        
        # Prepare evaluation context
        evaluation_context = {
            "aggregated_items": aggregated_items,
            "document_info": document_info or {},
            "user_preferences": user_preferences or {},
            "evaluation_criteria": evaluation_criteria,
            "rating_field": self._get_rating_field_name()
        }
        
        # Execute evaluation
        try:
            result = await self.execute_task(evaluation_context)
            
            # Organize evaluated items by severity/priority
            organized_result = self._organize_evaluated_items(result)
            
            # Generate executive summary if not already present
            if "executive_summary" not in organized_result:
                organized_result["executive_summary"] = self._generate_executive_summary(organized_result)
            
            return organized_result
            
        except Exception as e:
            logger.error(f"Error in evaluate_items: {e}")
            
            # Create fallback evaluation in case of error
            return self._create_fallback_evaluation(aggregated_items)
    
    def _get_stage_specific_content(self, context) -> str:
        """
        Get stage-specific content for the prompt.
        
        Args:
            context: Evaluation context
            
        Returns:
            Stage-specific content string
        """
        if isinstance(context, dict) and "aggregated_items" in context:
            # Build evaluation prompt
            content = ["EVALUATION GUIDELINES:"]
            
            # Add criteria information
            criteria = context.get("evaluation_criteria", {})
            if criteria:
                content.append("Evaluation Criteria:")
                for level, description in criteria.items():
                    content.append(f"- {level}: {description}")
            
            # Add rating field
            rating_field = context.get("rating_field", "severity")
            content.append(f"\nRating Field: {rating_field}")
            
            # Add user preferences if available
            user_prefs = context.get("user_preferences", {})
            if user_prefs:
                content.append("\nUser Preferences:")
                
                # Add detail level
                detail_level = user_prefs.get("detail_level", "standard")
                content.append(f"Detail Level: {detail_level}")
                
                # Add focus areas
                focus_areas = user_prefs.get("focus_areas", [])
                if focus_areas:
                    content.append(f"Focus Areas: {', '.join(focus_areas)}")
                
                # Add custom instructions
                custom_instructions = user_prefs.get("user_instructions", "")
                if custom_instructions:
                    content.append(f"Custom Instructions: {custom_instructions}")
            
            # Add evaluation tasks
            content.append("\nEVALUATION TASKS:")
            content.append("1. Assess each item for severity/priority using the criteria above")
            content.append("2. Add an impact assessment for each item")
            content.append("3. Group items by severity/priority")
            content.append("4. Create an executive summary highlighting the most important findings")
            
            if self.crew_type == "issues":
                content.append("\nFor issues specifically:")
                content.append("- Verify that each issue has an appropriate severity rating")
                content.append("- Add an impact assessment describing potential consequences")
                content.append("- Group issues into critical, high, medium, and low categories")
                content.append("- Create an executive summary focused on critical and high issues")
            
            # Add items to evaluate
            content.append("\nITEMS TO EVALUATE:")
            
            # Get aggregated items
            items = []
            input_field = self._get_input_field_name()
            if input_field in context["aggregated_items"]:
                items = context["aggregated_items"][input_field]
            
            # Format items in a concise way
            formatted_items = self._format_items_for_prompt(items)
            content.append(formatted_items)
            
            # Specify output format
            content.append("\nOUTPUT FORMAT:")
            
            if self.crew_type == "issues":
                content.append("Provide your evaluation as JSON with these keys:")
                content.append("- executive_summary: A summary of the key issues")
                content.append("- critical_issues: List of issues with critical severity")
                content.append("- high_issues: List of issues with high severity")
                content.append("- medium_issues: List of issues with medium severity")
                content.append("- low_issues: List of issues with low severity")
            else:
                content.append("Provide your evaluation as JSON with these keys:")
                content.append("- executive_summary: A summary of the key items")
                content.append(f"- evaluated_{self.crew_type}: List of evaluated items")
            
            return "\n".join(content)
        
        return ""
    
    def _format_items_for_prompt(self, items: List[Dict[str, Any]]) -> str:
        """
        Format items in a concise way for the prompt.
        
        Args:
            items: List of items to format
            
        Returns:
            Formatted items string
        """
        if not items:
            return "No items to evaluate."
        
        # For a small number of items, include full details
        if len(items) <= 15:
            return json.dumps(items, indent=2)
        
        # For more items, create a more compact representation
        compact_items = []
        
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
                
            compact_item = {
                "id": i + 1,
                "title": item.get("title", "Untitled Item")
            }
            
            # Add truncated description
            desc = item.get("description", "")
            if len(desc) > 100:
                desc = desc[:97] + "..."
            compact_item["description"] = desc
            
            # Add other important fields
            for field in ["category", "severity", "mention_count"]:
                if field in item:
                    compact_item[field] = item[field]
            
            compact_items.append(compact_item)
        
        return json.dumps(compact_items, indent=1)
    
    def _get_evaluation_criteria(self) -> Dict[str, str]:
        """
        Get evaluation criteria based on crew type.
        
        Returns:
            Dictionary of criteria
        """
        # First try to get from config
        if self.crew_type == "issues":
            severity_levels = self.config.get("issue_definition", {}).get("severity_levels", {})
            if severity_levels:
                return severity_levels
        
        # Default criteria if not in config
        defaults = {
            "issues": {
                "critical": "Immediate threat requiring urgent attention",
                "high": "Significant impact requiring prompt attention",
                "medium": "Moderate impact that should be addressed",
                "low": "Minor impact with limited consequences"
            },
            "actions": {
                "high": "Essential for success, must be completed",
                "medium": "Important but not critical",
                "low": "Helpful but optional"
            },
            "opportunities": {
                "high": "Significant potential benefit, strong ROI",
                "medium": "Moderate potential benefit, good ROI",
                "low": "Limited potential benefit, lower ROI"
            },
            "risks": {
                "critical": "High probability and major impact",
                "high": "High probability or major impact",
                "medium": "Moderate probability and impact",
                "low": "Low probability and limited impact"
            }
        }
        
        return defaults.get(self.crew_type, {"high": "Important", "medium": "Moderate", "low": "Minor"})
    
    def _get_rating_field_name(self) -> str:
        """
        Get the name of the rating field based on crew type.
        
        Returns:
            Rating field name
        """
        # Map crew types to rating field names
        field_map = {
            "issues": "severity",
            "actions": "priority",
            "opportunities": "value",
            "risks": "risk_level"
        }
        
        return field_map.get(self.crew_type, "rating")
    
    def _get_input_field_name(self) -> str:
        """
        Get the field name for input aggregated items.
        
        Returns:
            Field name for input items
        """
        # Map crew types to input field names
        field_map = {
            "issues": "aggregated_issues",
            "actions": "aggregated_action_items",
            "opportunities": "aggregated_opportunities",
            "risks": "aggregated_risks"
        }
        
        return field_map.get(self.crew_type, f"aggregated_{self.crew_type}_items")
    
    def _organize_evaluated_items(self, result: Any) -> Dict[str, Any]:
        """
        Organize evaluated items by severity/priority.
        
        Args:
            result: Evaluation result
            
        Returns:
            Organized result
        """
        # Parse the result if it's a string
        if isinstance(result, str):
            try:
                parsed_result = self.parse_llm_json(result)
                if isinstance(parsed_result, dict):
                    result = parsed_result
                else:
                    return self._create_empty_result()
            except Exception:
                return self._create_empty_result()
        
        # If result is not a dictionary, create empty result
        if not isinstance(result, dict):
            return self._create_empty_result()
        
        # For issues, organize by severity
        if self.crew_type == "issues":
            # Check if already organized by severity
            if all(f"{level}_issues" in result for level in ["critical", "high", "medium", "low"]):
                return result
            
            # Organize into severity groups
            organized = {
                "critical_issues": [],
                "high_issues": [],
                "medium_issues": [],
                "low_issues": []
            }
            
            # Get executive summary if available
            if "executive_summary" in result:
                organized["executive_summary"] = result["executive_summary"]
            
            # Look for evaluated issues
            evaluated_issues = None
            if "evaluated_issues" in result:
                evaluated_issues = result["evaluated_issues"]
            elif any(key.startswith("evaluated_") for key in result.keys()):
                for key in result.keys():
                    if key.startswith("evaluated_"):
                        evaluated_issues = result[key]
                        break
            
            # Group by severity
            if evaluated_issues and isinstance(evaluated_issues, list):
                for issue in evaluated_issues:
                    if not isinstance(issue, dict):
                        continue
                        
                    # Get severity
                    severity = issue.get("severity", "medium").lower()
                    
                    # Map to one of our standard levels
                    if severity in ["critical", "high", "medium", "low"]:
                        level = severity
                    elif severity in ["severe", "urgent", "important"]:
                        level = "critical"
                    elif severity in ["major", "significant"]:
                        level = "high"
                    elif severity in ["minor", "low", "trivial"]:
                        level = "low"
                    else:
                        level = "medium"
                    
                    # Add to appropriate group
                    organized[f"{level}_issues"].append(issue)
            
            return organized
            
        # For other types, just pass through
        return result
    
    def _generate_executive_summary(self, organized_result: Dict[str, Any]) -> str:
        """
        Generate an executive summary for the evaluation.
        
        Args:
            organized_result: Organized evaluation result
            
        Returns:
            Executive summary string
        """
        if self.crew_type == "issues":
            # Count issues by severity
            critical_count = len(organized_result.get("critical_issues", []))
            high_count = len(organized_result.get("high_issues", []))
            medium_count = len(organized_result.get("medium_issues", []))
            low_count = len(organized_result.get("low_issues", []))
            total_count = critical_count + high_count + medium_count + low_count
            
            if total_count == 0:
                return "No issues were identified in this document."
                
            # Create summary
            summary = f"Analysis identified {total_count} total issues: "
            summary += f"{critical_count} critical, {high_count} high, "
            summary += f"{medium_count} medium, and {low_count} low priority. "
            
            # Add details about critical issues
            if critical_count > 0:
                critical_issues = organized_result.get("critical_issues", [])
                critical_titles = [issue.get("title", "Untitled Issue") for issue in critical_issues[:3]]
                
                if critical_count <= 3:
                    summary += f"Critical issues include: {', '.join(critical_titles)}. "
                else:
                    summary += f"Critical issues include: {', '.join(critical_titles)}, and {critical_count - 3} more. "
            
            # Add details about high issues
            if high_count > 0 and critical_count < 2:
                high_issues = organized_result.get("high_issues", [])
                high_titles = [issue.get("title", "Untitled Issue") for issue in high_issues[:2]]
                
                if high_count <= 2:
                    summary += f"High priority issues include: {', '.join(high_titles)}."
                else:
                    summary += f"High priority issues include: {', '.join(high_titles)}, and {high_count - 2} more."
            
            return summary
            
        # Generic summary for other types
        return f"Analysis completed for {self.crew_type}."
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """
        Create an empty result when no items to evaluate.
        
        Returns:
            Empty result structure
        """
        if self.crew_type == "issues":
            return {
                "executive_summary": "No issues were identified in this document.",
                "critical_issues": [],
                "high_issues": [],
                "medium_issues": [],
                "low_issues": []
            }
        else:
            return {
                "executive_summary": f"No {self.crew_type} were identified in this document.",
                f"evaluated_{self.crew_type}": []
            }
    
    def _create_fallback_evaluation(self, aggregated_items: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a simple evaluation when LLM-based evaluation fails.
        
        Args:
            aggregated_items: Aggregated items
            
        Returns:
            Simple evaluation result
        """
        logger.info("Creating fallback evaluation")
        
        # Initialize result
        if self.crew_type == "issues":
            result = {
                "executive_summary": "",
                "critical_issues": [],
                "high_issues": [],
                "medium_issues": [],
                "low_issues": []
            }
        else:
            result = {
                "executive_summary": "",
                f"evaluated_{self.crew_type}": []
            }
        
        # Get items
        items = []
        input_field = self._get_input_field_name()
        if input_field in aggregated_items:
            items = aggregated_items[input_field]
        
        # If no items, return empty result
        if not items:
            return self._create_empty_result()
        
        # For issues, group by severity
        if self.crew_type == "issues":
            for item in items:
                if not isinstance(item, dict):
                    continue
                    
                # Copy item for evaluation
                evaluated_item = item.copy()
                
                # Get severity
                severity = evaluated_item.get("severity", "").lower()
                
                # Map to standard levels
                if severity in ["critical", "high", "medium", "low"]:
                    level = severity
                elif severity in ["severe", "urgent", "important"]:
                    level = "critical"
                elif severity in ["major", "significant"]:
                    level = "high"
                elif severity in ["minor", "low", "trivial"]:
                    level = "low"
                else:
                    # Default to medium
                    level = "medium"
                    evaluated_item["severity"] = "medium"
                
                # Add impact assessment if not present
                if "impact" not in evaluated_item:
                    evaluated_item["impact"] = self._generate_simple_impact(evaluated_item)
                
                # Add to appropriate group
                result[f"{level}_issues"].append(evaluated_item)
            
            # Generate executive summary
            result["executive_summary"] = self._generate_executive_summary(result)
            
            return result
        else:
            # For other types, simply pass through items with minimal processing
            evaluated_items = []
            
            for item in items:
                if not isinstance(item, dict):
                    continue
                    
                # Copy item for evaluation
                evaluated_item = item.copy()
                
                # Add rating if not present
                rating_field = self._get_rating_field_name()
                if rating_field not in evaluated_item:
                    evaluated_item[rating_field] = "medium"
                
                evaluated_items.append(evaluated_item)
            
            # Set evaluated items
            result[f"evaluated_{self.crew_type}"] = evaluated_items
            
            # Generate simple summary
            result["executive_summary"] = f"Analysis identified {len(evaluated_items)} {self.crew_type}."
            
            return result
    
    def _generate_simple_impact(self, item: Dict[str, Any]) -> str:
        """
        Generate a simple impact assessment for an item.
        
        Args:
            item: Item to generate impact for
            
        Returns:
            Simple impact assessment
        """
        # For issues, generate based on severity
        severity = item.get("severity", "medium").lower()
        
        if severity == "critical":
            return "Immediate negative impact on objectives if not addressed."
        elif severity == "high":
            return "Significant impact on project outcomes or performance."
        elif severity == "medium":
            return "Moderate impact on efficiency or effectiveness."
        elif severity == "low":
            return "Minor impact with limited consequences."
        else:
            return "Impact requires assessment."