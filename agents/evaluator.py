# agents/evaluator.py
"""
Evaluator Agent - Specialized in assessing importance and impact of identified items.
Works with ProcessingContext and integrated crew architecture.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base import BaseAgent

logger = logging.getLogger(__name__)

class EvaluatorAgent(BaseAgent):
    """
    Agent specialized in evaluating the importance, severity, or impact of identified items.
    Enhances metadata with assessment information and works with ProcessingContext.
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
            agent_type="evaluation",
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
        
        # Evaluate the results
        evaluated_result = self.evaluate_items(
            aggregated_items=aggregated_result,
            document_info=context.document_info
        )
        
        return evaluated_result
    
    # In agents/evaluator.py:
    async def evaluate_items(
        self, 
        aggregated_items: Dict[str, Any], 
        document_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate aggregated items for severity, importance, or impact.
        Adds assessment metadata to each item.
        
        Args:
            aggregated_items: Items to evaluate
            document_info: Optional document metadata
            
        Returns:
            Evaluated items with ratings, rationales, and enhanced metadata
        """
        # Get evaluation criteria from config
        criteria = self._get_evaluation_criteria()
        
        # Prepare context for prompt building
        context = {
            "aggregated_items": aggregated_items,
            "document_info": document_info or {},
            "evaluation_criteria": criteria
        }
        
        # Add rating scale information
        context["rating_scale"] = self._get_rating_scale_description()
        
        # Execute the evaluation task - ADD AWAIT HERE
        result = await self.execute_task(context=context)
        
        # Enhance the result with evaluation metadata
        result = self._enhance_result_with_metadata(result, aggregated_items)
        
        return result
    
    def _get_stage_specific_content(self, context) -> str:
        """Get stage-specific content for the prompt."""
        # If context is a dictionary with aggregated_items
        if isinstance(context, dict) and "aggregated_items" in context:
            aggregated_items = context["aggregated_items"]
            
            # Get evaluation criteria if available
            criteria_str = ""
            if "evaluation_criteria" in context:
                criteria = context["evaluation_criteria"]
                criteria_str = "\n".join([f"- {level}: {desc}" for level, desc in criteria.items()])
            
            # Get rating scale if available
            rating_scale = context.get("rating_scale", "")
            
            # Format aggregated items
            items_summary = json.dumps(aggregated_items, indent=2, default=str)
            
            # Add truncation if too long
            if len(items_summary) > 3000:
                items_summary = items_summary[:3000] + "...\n[Output truncated]"
            
            return f"""
            AGGREGATED ITEMS:
            {items_summary}
            
            EVALUATION CRITERIA:
            {criteria_str}
            
            RATING SCALE:
            {rating_scale}
            """
        
        # Otherwise, return empty string
        return ""
    
    def _get_evaluation_criteria(self) -> Dict[str, str]:
        """
        Get evaluation criteria from config based on crew type.
        
        Returns:
            Dictionary of evaluation criteria
        """
        # Try to get criteria from config
        criteria = {}
        
        # Look for criteria in analysis_definition section
        if "analysis_definition" in self.config:
            definition = self.config["analysis_definition"]
            
            # Get severity/importance/priority levels
            for levels_key in ["severity_levels", "importance_levels", "priority_levels"]:
                if levels_key in definition:
                    criteria = definition[levels_key]
                    break
        
        # If no criteria found, use defaults
        if not criteria:
            criteria = {
                "high": f"Significant impact on objectives or outcomes, requires immediate attention",
                "medium": f"Moderate impact that should be addressed but isn't urgent",
                "low": f"Minor impact, could be addressed as part of normal operations"
            }
            
            # Add critical level for issues
            if self.crew_type == "issues":
                criteria["critical"] = "Immediate threat or blockage that must be addressed immediately"
        
        return criteria
    
    def _get_rating_scale_description(self) -> str:
        """
        Get a description of the rating scale based on crew type.
        
        Returns:
            Description of the rating scale
        """
        # Default rating descriptions by crew type
        rating_descriptions = {
            "issues": "Severity rating indicates the potential negative impact if the issue is not addressed",
            "actions": "Priority rating indicates the urgency and importance of completing the action",
            "opportunities": "Value rating indicates the potential benefit if the opportunity is pursued",
            "risks": "Risk rating indicates the combination of probability and impact"
        }
        
        # Get appropriate description or default
        return rating_descriptions.get(self.crew_type, "Rating indicates the relative importance")
    
    def _enhance_result_with_metadata(self, result: Any, aggregated_items: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance evaluation result with additional metadata.
        
        Args:
            result: Evaluation result
            aggregated_items: Original aggregated items
            
        Returns:
            Enhanced result with metadata
        """
        # Handle string results
        if isinstance(result, str):
            try:
                # Try to parse as JSON
                parsed_result = json.loads(result)
                if isinstance(parsed_result, dict):
                    result = parsed_result
            except json.JSONDecodeError:
                # Not JSON, convert to basic dictionary
                key_field = self.get_key_field()
                result = {
                    key_field: [{"description": result}]
                }
        
        # Handle non-dictionary results
        if not isinstance(result, dict):
            key_field = self.get_key_field()
            result = {
                key_field: [{"description": str(result)}]
            }
        
        # Ensure the key field exists
        key_field = self.get_key_field()
        if key_field not in result:
            result[key_field] = []
        
        # Analyze ratings distribution
        items = result[key_field]
        rating_distribution = {}
        
        # Get input items to map metadata
        input_key_field = self._get_input_key_field()
        input_items = []
        if isinstance(aggregated_items, dict) and input_key_field in aggregated_items:
            input_items = aggregated_items[input_key_field]
            if not isinstance(input_items, list):
                input_items = []
        
        # Create lookup for input items
        input_lookup = {}
        for item in input_items:
            if isinstance(item, dict):
                item_title = item.get("title", "")
                if item_title:
                    input_lookup[item_title] = item
        
        if isinstance(items, list):
            # Analyze rating distribution
            for item in items:
                if not isinstance(item, dict):
                    continue
                
                # Get the rating field name based on crew type
                rating_field = self._get_rating_field_name()
                
                if rating_field in item:
                    rating = item[rating_field]
                    if rating not in rating_distribution:
                        rating_distribution[rating] = 0
                    rating_distribution[rating] += 1
                
                # Transfer metadata from input item
                title = item.get("title", "")
                if title and title in input_lookup:
                    input_item = input_lookup[title]
                    
                    # Copy metadata that shouldn't change during evaluation
                    for field in ["mention_count", "source_chunks", "keywords"]:
                        if field in input_item and field not in item:
                            item[field] = input_item[field]
                
                # Add priority if not present (numeric value based on rating)
                if "priority" not in item and rating_field in item:
                    item["priority"] = self._rating_to_priority(item[rating_field])
                
                # Add impact assessment if not present
                if "impact_assessment" not in item and "description" in item and rating_field in item:
                    item["impact_assessment"] = self._generate_impact_assessment(
                        item["description"], 
                        item[rating_field]
                    )
                
                # Add related items if not present
                if "related_items" not in item:
                    item["related_items"] = []
        
        # Add evaluation metadata
        result["_metadata"] = {
            "evaluated_count": len(items) if isinstance(items, list) else 0,
            "rating_distribution": rating_distribution,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def get_key_field(self) -> str:
        """
        Get the key field name for the evaluated items based on crew type.
        
        Returns:
            Field name for evaluated items
        """
        # Default mapping
        field_mapping = {
            "issues": "evaluated_issues",
            "actions": "evaluated_actions",
            "opportunities": "evaluated_opportunities",
            "risks": "evaluated_risks"
        }
        
        # Get from config if available
        return field_mapping.get(self.crew_type, f"evaluated_{self.crew_type}_items")
    
    def _get_input_key_field(self) -> str:
        """
        Get the key field name for the input aggregated items.
        
        Returns:
            Field name for input items
        """
        # Default mapping
        field_mapping = {
            "issues": "aggregated_issues",
            "actions": "aggregated_action_items",
            "opportunities": "aggregated_opportunities",
            "risks": "aggregated_risks"
        }
        
        # Get from config if available
        return field_mapping.get(self.crew_type, f"aggregated_{self.crew_type}_items")
    
    def _get_rating_field_name(self) -> str:
        """
        Get the name of the rating field based on crew type.
        
        Returns:
            Name of the rating field
        """
        # Default mapping
        field_mapping = {
            "issues": "severity",
            "actions": "priority",
            "opportunities": "value",
            "risks": "risk_level"
        }
        
        # Get from config if available
        return field_mapping.get(self.crew_type, "rating")
    
    def _rating_to_priority(self, rating: str) -> int:
        """
        Convert a rating to a numeric priority.
        
        Args:
            rating: Rating string (e.g., "critical", "high", "medium", "low")
            
        Returns:
            Numeric priority (lower is higher priority)
        """
        # Map ratings to numeric priorities (lower numbers = higher priority)
        priority_map = {
            "critical": 1,
            "high": 2,
            "medium": 3,
            "low": 4
        }
        
        return priority_map.get(rating.lower(), 3)
    
    def _generate_impact_assessment(self, description: str, rating: str) -> str:
        """
        Generate a simple impact assessment based on the description and rating.
        
        Args:
            description: Item description
            rating: Item rating
            
        Returns:
            Impact assessment text
        """
        # Generate a simple impact assessment based on rating
        if rating.lower() == "critical":
            return f"Severe impact that could significantly damage objectives or operations if not addressed immediately."
        elif rating.lower() == "high":
            return f"Substantial impact requiring prompt attention to prevent significant negative consequences."
        elif rating.lower() == "medium":
            return f"Moderate impact that should be addressed to improve effectiveness and efficiency."
        elif rating.lower() == "low":
            return f"Minor impact that could be addressed as part of normal improvements."
        else:
            return f"Impact level is unclear and requires further assessment."