"""
Formatter Agent - Specialized in creating structured reports from analysis results.
Clean implementation that leverages the new BaseAgent architecture.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base import BaseAgent

logger = logging.getLogger(__name__)

class FormatterAgent(BaseAgent):
    """
    Agent specialized in formatting analysis results into a clear, structured report.
    Uses metadata to enhance organization and presentation.
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
            agent_type="formatter",
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
        formatted_result = await self.format_report(
            evaluated_items=evaluated_result,
            document_info=context.document_info,
            user_preferences=context.options
        )
        
        return formatted_result
    
    async def format_report(
        self, 
        evaluated_items: Dict[str, Any], 
        document_info: Optional[Dict[str, Any]] = None, 
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format evaluated items into a structured report.
        
        Args:
            evaluated_items: Items to include in the report
            document_info: Optional document metadata
            user_preferences: Optional user formatting preferences
            
        Returns:
            Formatted report
        """
        # Get report format information
        report_format = self._get_report_format()
        
        # Prepare formatting context
        formatting_context = {
            "evaluated_items": evaluated_items,
            "document_info": document_info or {},
            "user_preferences": user_preferences or {},
            "report_format": report_format,
            "detail_level": user_preferences.get("detail_level", "standard") if user_preferences else "standard"
        }
        
        # Get template for this report type
        template = self._get_template()
        if template:
            formatting_context["template"] = template
        
        # Execute formatting
        result = await self.execute_task(formatting_context)
        
        # Handle HTML result
        if isinstance(result, str) and self._is_html_content(result):
            # Add metadata as HTML comment
            result = self._add_metadata_to_html(result, evaluated_items, user_preferences)
            return result
        
        # Handle other result types
        return self._add_metadata_to_result(result, evaluated_items, user_preferences)
    
    def _get_stage_specific_content(self, context) -> str:
        """Get stage-specific content for the prompt."""
        if isinstance(context, dict):
            content_parts = []
            
            # Add template if available
            if "template" in context:
                template = context["template"]
                # Truncate if very long
                if len(template) > 1000:
                    template = template[:1000] + "...(template truncated)..."
                content_parts.append(f"REPORT TEMPLATE:\n{template}")
            
            # Add report format guidance
            if "report_format" in context:
                report_format = context["report_format"]
                
                # Include sections
                if "sections" in report_format:
                    sections = report_format["sections"]
                    content_parts.append(f"REPORT SECTIONS:\n- " + "\n- ".join(sections))
                
                # Include item format
                if "issue_presentation" in report_format:
                    presentation = report_format["issue_presentation"]
                    presentation_str = json.dumps(presentation, indent=2)
                    content_parts.append(f"ITEM PRESENTATION FORMAT:\n{presentation_str}")
            
            # Add detail level guidance
            detail_level = context.get("detail_level", "standard")
            detail_guidance = self._get_detail_level_guidance(detail_level)
            content_parts.append(f"DETAIL LEVEL ({detail_level}):\n{detail_guidance}")
            
            # Add items to format
            if "evaluated_items" in context:
                evaluated_items = context["evaluated_items"]
                
                # Find the input field name
                input_field = self._get_input_field_name()
                
                if input_field in evaluated_items:
                    items = evaluated_items[input_field]
                    
                    # Add distribution statistics
                    rating_field = self._get_rating_field_name()
                    rating_counts = {}
                    
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict) and rating_field in item:
                                rating = item[rating_field]
                                if rating not in rating_counts:
                                    rating_counts[rating] = 0
                                rating_counts[rating] += 1
                    
                    if rating_counts:
                        count_str = ", ".join([f"{rating}: {count}" for rating, count in rating_counts.items()])
                        content_parts.append(f"ITEM COUNTS:\n{count_str}")
                    
                    # Add sample items (truncated if needed)
                    item_samples = []
                    if isinstance(items, list):
                        sample_count = min(3, len(items))
                        for i in range(sample_count):
                            item_str = json.dumps(items[i], indent=2)
                            if len(item_str) > 500:
                                item_str = item_str[:500] + "...(truncated)..."
                            item_samples.append(item_str)
                    
                    if item_samples:
                        content_parts.append(f"SAMPLE ITEMS:\n" + "\n\n".join(item_samples))
            
            return "\n\n".join(content_parts)
        
        return ""
    
    def _get_template(self) -> str:
        """
        Get the HTML template for the report.
        
        Returns:
            HTML template string
        """
        # Try to get from config
        template = self.config.get("report_format", {}).get("html_template", "")
        
        if not template:
            # Use default template based on crew type
            if self.crew_type == "issues":
                return self._get_default_issues_template()
            elif self.crew_type == "actions":
                return self._get_default_actions_template()
            else:
                return self._get_default_template()
        
        return template
    
    def _get_report_format(self) -> Dict[str, Any]:
        """
        Get the report format configuration.
        
        Returns:
            Report format dictionary
        """
        # Try to get from config
        report_format = self.config.get("report_format", {})
        
        if not report_format:
            # Use default format based on crew type
            if self.crew_type == "issues":
                return {
                    "sections": [
                        "Executive Summary",
                        "Critical Issues",
                        "High-Priority Issues",
                        "Medium-Priority Issues",
                        "Low-Priority Issues"
                    ],
                    "issue_presentation": {
                        "title": "Clear, descriptive title",
                        "severity": "Visual indicator of severity",
                        "description": "Full issue description",
                        "impact": "Potential consequences",
                        "category": "Issue category"
                    }
                }
            elif self.crew_type == "actions":
                return {
                    "sections": [
                        "Executive Summary",
                        "Immediate Actions",
                        "Short-Term Actions",
                        "Long-Term Actions"
                    ],
                    "issue_presentation": {
                        "title": "Action title",
                        "priority": "Action priority",
                        "description": "Detailed description",
                        "owner": "Assigned to",
                        "due_date": "Timeframe or deadline"
                    }
                }
            else:
                return {
                    "sections": [
                        "Executive Summary",
                        "Key Findings",
                        "Detailed Analysis"
                    ],
                    "issue_presentation": {
                        "title": "Item title",
                        "description": "Detailed description",
                        "importance": "Level of importance"
                    }
                }
        
        return report_format
    
    def _get_detail_level_guidance(self, detail_level: str) -> str:
        """
        Get guidance for a specific detail level.
        
        Args:
            detail_level: Detail level (essential, standard, comprehensive)
            
        Returns:
            Guidance string
        """
        # Try to get from config
        user_options = self.config.get("user_options", {})
        detail_levels = user_options.get("detail_levels", {})
        
        if detail_level in detail_levels:
            return detail_levels[detail_level]
        
        # Default guidance
        defaults = {
            "essential": "Focus only on the most important elements with minimal detail.",
            "standard": "Provide a balanced amount of detail, covering all significant aspects.",
            "comprehensive": "Include thorough details, context, and explanations for all elements."
        }
        
        return defaults.get(detail_level, defaults["standard"])
    
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
    
    def _add_metadata_to_html(self, html: str, evaluated_items: Dict[str, Any], user_preferences: Optional[Dict[str, Any]]) -> str:
        """
        Add metadata to HTML as a comment.
        
        Args:
            html: HTML content
            evaluated_items: Evaluated items
            user_preferences: User preferences
            
        Returns:
            HTML with metadata comment
        """
        # Create metadata JSON
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "crew_type": self.crew_type,
            "detail_level": user_preferences.get("detail_level", "standard") if user_preferences else "standard",
            "focus_areas": user_preferences.get("focus_areas", []) if user_preferences else []
        }
        
        # Add rating distribution if available
        if isinstance(evaluated_items, dict) and "_metadata" in evaluated_items:
            eval_metadata = evaluated_items["_metadata"]
            if "rating_distribution" in eval_metadata:
                metadata["rating_distribution"] = eval_metadata["rating_distribution"]
        
        # Format metadata as JSON
        metadata_json = json.dumps(metadata, indent=2)
        
        # Add as HTML comment at the end of document
        if html.lower().endswith("</html>"):
            # Insert before closing HTML tag
            result = html[:-7] + f"\n\n<!-- REPORT METADATA\n{metadata_json}\n-->\n</html>"
        else:
            # Append to the end
            result = html + f"\n\n<!-- REPORT METADATA\n{metadata_json}\n-->"
        
        return result    
    
    def _get_default_issues_template(self) -> str:
        """
        Get default HTML template for issues report.
        Uses class names that match the CSS in core_styling.py.
        """
        return """<div class="issues-report">
    <h1>Issues Analysis Report</h1>

    <div class="executive-summary">
        <h2>📋 Executive Summary</h2>
        <p>[Executive summary content]</p>
    </div>

    <div class="issues-section critical-section">
        <h2>🔴 Critical Issues</h2>
        [Critical issues content]
    </div>

    <div class="issues-section high-section">
        <h2>🟠 High-Priority Issues</h2>
        [High priority issues content]
    </div>

    <div class="issues-section medium-section">
        <h2>🟡 Medium-Priority Issues</h2>
        [Medium priority issues content]
    </div>

    <div class="issues-section low-section">
        <h2>🟢 Low-Priority Issues</h2>
        [Low priority issues content]
    </div>

    <div class="summary-stats">
        <h2>📊 Summary Statistics</h2>
        [Summary statistics content]
    </div>
    </div>"""

    def _get_default_actions_template(self) -> str:
        """
        Get default HTML template for actions report.
        
        Returns:
            HTML template string
        """
        return """<div class="actions-report">
    <h1>Action Items Report</h1>

    <div class="executive-summary">
        <h2>📋 Executive Summary</h2>
        <p>[Executive summary content]</p>
    </div>

    <div class="actions-section immediate-section">
        <h2>🔴 Immediate Actions</h2>
        [Immediate actions content]
    </div>

    <div class="actions-section short-term-section">
        <h2>🟠 Short-Term Actions</h2>
        [Short-term actions content]
    </div>

    <div class="actions-section long-term-section">
        <h2>🟢 Long-Term Actions</h2>
        [Long-term actions content]
    </div>

    <div class="summary-stats">
        <h2>📊 Summary Statistics</h2>
        [Summary statistics content]
    </div>
    </div>"""

    def _get_default_template(self) -> str:
        """
        Get default HTML template for general reports.
        
        Returns:
            HTML template string
        """
        return """<div class="analysis-report">
    <h1>Analysis Report</h1>

    <div class="executive-summary">
        <h2>📋 Executive Summary</h2>
        <p>[Executive summary content]</p>
    </div>

    <div class="findings-section">
        <h2>🔍 Key Findings</h2>
        [Key findings content]
    </div>

    <div class="details-section">
        <h2>📊 Detailed Analysis</h2>
        [Detailed analysis content]
    </div>
    </div>"""

    def _get_input_field_name(self) -> str:
        """
        Get the field name for input evaluated items.
        
        Returns:
            Field name for input items
        """
        # Map crew types to input field names
        field_map = {
            "issues": "evaluated_issues",
            "actions": "evaluated_actions",
            "opportunities": "evaluated_opportunities",
            "risks": "evaluated_risks"
        }
        
        return field_map.get(self.crew_type, f"evaluated_{self.crew_type}_items")

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