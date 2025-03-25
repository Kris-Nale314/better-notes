"""
Evaluator Agent - Specialized in assessing importance and impact of identified items.
Clean implementation that leverages the new BaseAgent architecture.
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
    Agent specialized in evaluating the importance, severity, or impact of identified items.
    Enhances metadata with assessment information.
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
    
    async def process(self, context):
        """
        Process aggregated results using the context.
        
        Args:
            context: ProcessingContext object
            
        Returns:
            Evaluated results
        """
        # Get aggregation results from context
        aggregated_result = context.results.get("aggregation", {})
        
        # Evaluate results
        evaluated_result = await self.evaluate_items(
            aggregated_items=aggregated_result,
            document_info=context.document_info
        )
        
        return evaluated_result
    
    async def evaluate_items(
        self, 
        aggregated_items: Dict[str, Any], 
        document_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate aggregated items for severity, importance, or impact.
        
        Args:
            aggregated_items: Items to evaluate
            document_info: Optional document metadata
            
        Returns:
            Evaluated items with ratings and rationales
        """
        # Get evaluation criteria
        evaluation_criteria = self._get_evaluation_criteria()
        
        # Prepare evaluation context
        evaluation_context = {
            "aggregated_items": aggregated_items,
            "document_info": document_info or {},
            "evaluation_criteria": evaluation_criteria,
            "rating_field": self._get_rating_field_name()
        }
        
        # Execute evaluation
        result = await self.execute_task(evaluation_context)
        
        # Enhance result with metadata
        result = self._enhance_result_with_metadata(result, aggregated_items)
        
        return result
    
    def _get_stage_specific_content(self, context) -> str:
        """Get stage-specific content for the prompt."""
        if isinstance(context, dict) and "aggregated_items" in context:
            # Add evaluation criteria
            content = "EVALUATION CRITERIA:\n"
            
            # Add criteria information
            criteria = context.get("evaluation_criteria", {})
            for level, description in criteria.items():
                content += f"- {level}: {description}\n"
            
            content += f"\nRATING FIELD: {context.get('rating_field', 'severity')}\n\n"
            
            # Add items to evaluate
            aggregated_items = context.get("aggregated_items", {})
            
            # Find the key field
            input_field = None
            for key in aggregated_items.keys():
                if key.startswith("aggregated_"):
                    input_field = key
                    break
            
            if input_field and input_field in aggregated_items:
                items = aggregated_items[input_field]
                
                # Format items for the prompt
                items_json = json.dumps(items, indent=2)
                if len(items_json) > 3000:
                    # Truncate if too long
                    items_json = items_json[:3000] + "\n...(truncated for brevity)..."
                
                content += f"ITEMS TO EVALUATE:\n{items_json}\n"
            
            return content
        
        return ""
    
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
    
    def _get_output_field_name(self) -> str:
        """
        Get the field name for evaluated output items.
        
        Returns:
            Field name for output items
        """
        # Map crew types to output field names
        field_map = {
            "issues": "evaluated_issues",
            "actions": "evaluated_actions",
            "opportunities": "evaluated_opportunities",
            "risks": "evaluated_risks"
        }
        
        return field_map.get(self.crew_type, f"evaluated_{self.crew_type}_items")
    
    def _enhance_result_with_metadata(self, result: Any, aggregated_items: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance evaluation result with additional metadata.
        
        Args:
            result: Evaluation result
            aggregated_items: Original aggregated items
            
        Returns:
            Enhanced result with metadata
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
                    result = {self._get_output_field_name(): [{"description": result}]}
            except Exception:
                # Not valid JSON, create basic structure
                result = {self._get_output_field_name(): [{"description": result}]}
        
        # Handle non-dictionary results
        if not isinstance(result, dict):
            result = {self._get_output_field_name(): [{"description": str(result)}]}
        
        # Ensure output field exists
        output_field = self._get_output_field_name()
        if output_field not in result:
            result[output_field] = []
        
        # Get rating field name
        rating_field = self._get_rating_field_name()
        
        # Add original metadata from input items
        items = result[output_field]
        input_field = self._get_input_field_name()
        input_items = []
        
        if input_field in aggregated_items and isinstance(aggregated_items[input_field], list):
            input_items = aggregated_items[input_field]
        
        # Create a lookup for input items
        input_lookup = {}
        for item in input_items:
            if isinstance(item, dict) and "title" in item:
                input_lookup[item["title"]] = item
        
        # Transfer metadata and add missing fields
        if isinstance(items, list):
            # Track rating distribution
            rating_distribution = {}
            
            for item in items:
                if not isinstance(item, dict):
                    continue
                
                # Get the title to match with input items
                title = item.get("title", "")
                
                # Find matching input item
                if title in input_lookup:
                    input_item = input_lookup[title]
                    
                    # Copy metadata that shouldn't change during evaluation
                    for field in ["mention_count", "source_chunks", "keywords", "confidence"]:
                        if field in input_item and field not in item:
                            item[field] = input_item[field]
                
                # Add numeric priority if not present
                if "priority" not in item and rating_field in item:
                    item["priority"] = self._rating_to_priority(item[rating_field])
                
                # Add impact assessment if not present
                if "impact_assessment" not in item and "description" in item and rating_field in item:
                    item["impact_assessment"] = self._generate_impact_assessment(
                        item.get("description", ""),
                        item.get(rating_field, "")
                    )
                
                # Track rating distribution
                if rating_field in item:
                    rating = item[rating_field]
                    if rating not in rating_distribution:
                        rating_distribution[rating] = 0
                    rating_distribution[rating] += 1
        
        # Add evaluation metadata
        result["_metadata"] = {
            "evaluated_count": len(items) if isinstance(items, list) else 0,
            "rating_distribution": rating_distribution,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def _rating_to_priority(self, rating: str) -> int:
        """
        Convert a rating to a numeric priority.
        
        Args:
            rating: Rating string
            
        Returns:
            Numeric priority (lower is higher priority)
        """
        priority_map = {
            "critical": 1,
            "high": 2,
            "medium": 3,
            "low": 4
        }
        
        return priority_map.get(rating.lower(), 3)
    
    def _generate_impact_assessment(self, description: str, rating: str) -> str:
        """
        Generate a simple impact assessment based on description and rating.
        
        Args:
            description: Item description
            rating: Item rating
            
        Returns:
            Impact assessment text
        """
        # Basic assessments by rating level
        assessments = {
            "critical": "Critical impact that could severely damage objectives if not addressed immediately.",
            "high": "Significant impact requiring prompt attention to prevent negative outcomes.",
            "medium": "Moderate impact that should be addressed to improve effectiveness.",
            "low": "Minor impact that could be addressed as part of regular improvements."
        }
        
        return assessments.get(rating.lower(), "Impact level requires further assessment.")