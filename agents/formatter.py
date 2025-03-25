# agents/formatter.py
"""
Formatter Agent - Specialized in creating structured reports from analysis results.
Works with ProcessingContext and integrated crew architecture.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base import BaseAgent

logger = logging.getLogger(__name__)

class FormatterAgent(BaseAgent):
    """
    Agent specialized in formatting analysis results into a clear, structured report.
    Uses metadata to enhance organization and presentation.
    Works with ProcessingContext.
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
        """Initialize a formatter agent."""
        super().__init__(
            llm_client=llm_client,
            agent_type="formatting",
            crew_type=crew_type,
            config=config,
            config_manager=config_manager,
            verbose=verbose,
            max_chunk_size=max_chunk_size,
            max_rpm=max_rpm
        )
    
    async def process(self, context):
        """
        Process evaluated results using the context.
        
        Args:
            context: ProcessingContext object
            
        Returns:
            Formatted report
        """
        # Get evaluation results from context
        evaluated_result = context.results.get("evaluation", {})
        
        # Format the report
        formatted_result = self.format_report(
            evaluated_items=evaluated_result,
            document_info=context.document_info,
            user_preferences=context.options
        )
        
        return formatted_result
    
    # In agents/formatter.py:
    async def format_report(
        self, 
        evaluated_items: Dict[str, Any], 
        document_info: Optional[Dict[str, Any]] = None, 
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format evaluated items into a structured report.
        Uses metadata to enhance organization and presentation.
        
        Args:
            evaluated_items: Items to include in the report
            document_info: Optional document metadata
            user_preferences: Optional user formatting preferences
            
        Returns:
            Formatted report
        """
        # Get HTML template from config
        format_template = self.get_template()
        
        # Get report structure from config
        report_structure = self._get_report_structure()
        
        # Prepare context for prompt building
        context = {
            "evaluated_items": evaluated_items,
            "document_info": document_info or {},
            "user_preferences": user_preferences or {},
            "format_template": format_template,
            "report_structure": report_structure
        }
        
        # Add rating distribution for better context
        if isinstance(evaluated_items, dict) and "_metadata" in evaluated_items:
            metadata = evaluated_items["_metadata"]
            if "rating_distribution" in metadata:
                context["rating_distribution"] = metadata["rating_distribution"]
        
        # Add detail level guidance
        detail_level = user_preferences.get("detail_level", "standard") if user_preferences else "standard"
        context["detail_level"] = detail_level
        context["detail_guidance"] = self._get_detail_level_guidance(detail_level)
        
        # Execute the formatting task - ADD AWAIT HERE
        result = await self.execute_task(context=context)
        
        # Add formatting metadata to the result
        result = self._add_formatting_metadata(result, evaluated_items, user_preferences)
        
        return result
    
    def _get_stage_specific_content(self, context) -> str:
        """Get stage-specific content for the prompt."""
        # If context is a dictionary with evaluated_items
        if isinstance(context, dict):
            content_parts = []
            
            # Add evaluated items if available
            if "evaluated_items" in context:
                evaluated_items = context["evaluated_items"]
                items_summary = json.dumps(evaluated_items, indent=2, default=str)
                
                # Add truncation if too long
                if len(items_summary) > 3000:
                    items_summary = items_summary[:3000] + "...\n[Output truncated]"
                
                content_parts.append(f"EVALUATED ITEMS:\n{items_summary}")
            
            # Add template if available
            if "format_template" in context:
                template = context["format_template"]
                
                # Add truncation if too long
                if len(template) > 1000:
                    template = template[:1000] + "...\n[Template truncated]"
                
                content_parts.append(f"FORMAT TEMPLATE:\n{template}")
            
            # Add report structure if available
            if "report_structure" in context:
                structure = context["report_structure"]
                structure_str = json.dumps(structure, indent=2, default=str)
                
                content_parts.append(f"REPORT STRUCTURE:\n{structure_str}")
            
            # Add detail guidance if available
            if "detail_guidance" in context:
                content_parts.append(f"DETAIL LEVEL GUIDANCE:\n{context['detail_guidance']}")
            
            # Join all parts
            return "\n\n".join(content_parts)
        
        # Otherwise, return empty string
        return ""
    
    def get_template(self) -> str:
        """
        Get the template for formatting agents.
        
        Returns:
            Template string
        """
        if self.agent_type == "formatting":
            # First check new structure
            agent_config = self._get_agent_config()
            if "html_template" in agent_config:
                return agent_config["html_template"]
            
            # Fall back to old structure
            return self.config.get("formatting", {}).get("format_template", "")
        return ""
    
    def _get_report_structure(self) -> Dict[str, Any]:
        """
        Get the report structure from configuration.
        
        Returns:
            Report structure dictionary
        """
        # Get report structure from config
        report_structure = {}
        
        # Try to get from formatter config
        formatter_config = self.config.get("agents", {}).get("formatter", {})
        if "report_structure" in formatter_config:
            report_structure = formatter_config["report_structure"]
        
        # If no structure provided, create default
        if not report_structure:
            sections = self._get_default_sections()
            report_structure = {
                "sections": sections,
                "executive_summary_guidelines": "Focus on the most important findings and their implications"
            }
        
        return report_structure
    
    def _get_default_sections(self) -> List[str]:
        """
        Get the default report sections based on crew type.
        
        Returns:
            List of section names
        """
        common_sections = ["Executive Summary", "Methodology"]
        
        crew_specific_sections = {
            "issues": ["Critical Issues", "High-Priority Issues", "Medium-Priority Issues", "Low-Priority Issues"],
            "actions": ["Immediate Actions", "Short-Term Actions", "Long-Term Actions"],
            "opportunities": ["Strategic Opportunities", "Tactical Improvements", "Future Considerations"],
            "risks": ["High Risks", "Medium Risks", "Low Risks"]
        }
        
        return common_sections + crew_specific_sections.get(self.crew_type, ["Findings", "Analysis"])
    
    def _get_detail_level_guidance(self, detail_level: str) -> str:
            """
            Get guidance for the specified detail level.
            
            Args:
                detail_level: Detail level (essential, standard, comprehensive)
                
            Returns:
                Guidance string for the detail level
            """
            # Default detail level guidance
            guidance = {
                "essential": "Focus only on the most important elements with minimal detail. Prioritize brevity and clarity.",
                "standard": "Provide a balanced amount of detail, covering all significant aspects without excessive elaboration.",
                "comprehensive": "Include thorough details, context, and explanations for all elements in the report."
            }
            
            # Get from config if available
            config_guidance = self.config.get("user_options", {}).get("detail_levels", {}).get(detail_level, {}).get("description", "")
            
            return config_guidance if config_guidance else guidance.get(detail_level, "")
        
    def _add_formatting_metadata(
            self, 
            result: Any, 
            evaluated_items: Dict[str, Any],
            user_preferences: Optional[Dict[str, Any]]
        ) -> str:
            """
            Add formatting metadata to the report.
            
            Args:
                result: Formatting result
                evaluated_items: Input evaluated items
                user_preferences: User preferences
                
            Returns:
                Enhanced report with metadata
            """
            # Most formatter outputs will be HTML strings
            # If it's a string, just add metadata comment at the end
            if isinstance(result, str):
                # Get key statistics
                key_stats = self._extract_key_stats(evaluated_items)
                detail_level = user_preferences.get("detail_level", "standard") if user_preferences else "standard"
                focus_areas = user_preferences.get("focus_areas", []) if user_preferences else []
                
                # Create a metadata JSON for debugging
                metadata_json = json.dumps({
                    "generated_at": datetime.now().isoformat(),
                    "crew_type": self.crew_type,
                    "detail_level": detail_level,
                    "focus_areas": focus_areas,
                    "item_counts": key_stats
                }, indent=2)
                
                # Add as HTML comment at the end of the document
                if result.lower().endswith("</html>"):
                    result = result[:-7] + f"\n<!-- REPORT METADATA\n{metadata_json}\n-->\n</html>"
                else:
                    result = result + f"\n\n<!-- REPORT METADATA\n{metadata_json}\n-->"
                    
                return result
                
            # If it's a dictionary, add metadata inside
            elif isinstance(result, dict):
                result["_metadata"] = {
                    "generated_at": datetime.now().isoformat(),
                    "crew_type": self.crew_type,
                    "detail_level": user_preferences.get("detail_level", "standard") if user_preferences else "standard",
                    "focus_areas": user_preferences.get("focus_areas", []) if user_preferences else [],
                    "item_counts": self._extract_key_stats(evaluated_items)
                }
                
                return result
                
            # Return unchanged for other types
            return result
        
    def _extract_key_stats(self, evaluated_items: Dict[str, Any]) -> Dict[str, int]:
            """
            Extract key statistics from the evaluated items.
            
            Args:
                evaluated_items: Evaluated items
                
            Returns:
                Dictionary of key statistics
            """
            stats = {"total": 0}
            
            # Get the input key field
            key_field = self._get_input_key_field()
            
            if isinstance(evaluated_items, dict) and key_field in evaluated_items:
                items = evaluated_items[key_field]
                if isinstance(items, list):
                    stats["total"] = len(items)
                    
                    # Count by rating (severity, priority, etc.)
                    rating_counts = {}
                    rating_field = self._get_rating_field_name()
                    
                    for item in items:
                        if isinstance(item, dict) and rating_field in item:
                            rating = item[rating_field]
                            if rating not in rating_counts:
                                rating_counts[rating] = 0
                            rating_counts[rating] += 1
                    
                    stats.update(rating_counts)
            
            return stats
        
    def _get_input_key_field(self) -> str:
            """
            Get the key field name for the input evaluated items.
            
            Returns:
                Field name for input items
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