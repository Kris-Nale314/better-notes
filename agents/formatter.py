"""
FormatterAgent - Streamlit-compatible formatter for creating structured reports.
Completely rewritten to ensure proper HTML rendering in Streamlit.

This agent produces HTML reports that:
1. Render correctly in Streamlit using st.markdown with unsafe_allow_html=True
2. Match the CSS classes defined in core_styling.py
3. Handle issues, actions, and other report types
4. Provide consistent formatting and styling across all reports
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .base import BaseAgent

logger = logging.getLogger(__name__)

class FormatterAgent(BaseAgent):
    """
    Agent specialized in formatting analysis results into reports that render properly in Streamlit.
    Creates well-formed HTML that works with Streamlit's markdown rendering engine.
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
        """Initialize a Streamlit-optimized formatter agent."""
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
        Process evaluated results using the context and create a Streamlit-friendly report.
        
        Args:
            context: ProcessingContext object
            
        Returns:
            Formatted report as HTML string or structured dictionary
        """
        # Get evaluation results from context
        evaluated_result = context.results.get("evaluation", {})
        
        if not evaluated_result:
            logger.warning("No evaluation results found for formatting")
            return self._create_error_report("No evaluation results available")
        
        # Format the report based on crew type
        if self.crew_type == "issues":
            formatted_result = self._format_issues_report(evaluated_result, context)
        elif self.crew_type == "actions":
            formatted_result = self._format_actions_report(evaluated_result, context)
        else:
            formatted_result = self._format_generic_report(evaluated_result, context)
        
        return formatted_result
    
    def _format_issues_report(self, evaluated_result: Dict[str, Any], context) -> str:
        """
        Format issues data into a Streamlit-friendly HTML report.
        
        Args:
            evaluated_result: Evaluated issues data
            context: Processing context
            
        Returns:
            HTML report string
        """
        # Extract issues by severity
        critical_issues = evaluated_result.get("critical_issues", [])
        high_issues = evaluated_result.get("high_issues", [])
        medium_issues = evaluated_result.get("medium_issues", [])
        low_issues = evaluated_result.get("low_issues", [])
        
        # Get executive summary if available
        executive_summary = evaluated_result.get("executive_summary", 
                                                "No executive summary available.")
        
        # Get user preferences for display
        user_preferences = context.options if hasattr(context, "options") else {}
        detail_level = user_preferences.get("detail_level", "standard")
        
        # Build the HTML report
        html = [
            f'<div class="issues-report">',
            f'<h1>Issues Analysis Report</h1>',
            
            # Executive Summary section
            f'<div class="executive-summary">',
            f'<h2>📋 Executive Summary</h2>',
            f'<p>{executive_summary}</p>',
            f'</div>',
            
            # Critical Issues section
            f'<div class="issues-section">',
            f'<h2>🔴 Critical Issues ({len(critical_issues)})</h2>',
            self._render_issues_list(critical_issues, "critical", detail_level),
            f'</div>',
            
            # High Issues section
            f'<div class="issues-section">',
            f'<h2>🟠 High-Priority Issues ({len(high_issues)})</h2>',
            self._render_issues_list(high_issues, "high", detail_level),
            f'</div>',
            
            # Medium Issues section
            f'<div class="issues-section">',
            f'<h2>🟡 Medium-Priority Issues ({len(medium_issues)})</h2>',
            self._render_issues_list(medium_issues, "medium", detail_level),
            f'</div>',
            
            # Low Issues section
            f'<div class="issues-section">',
            f'<h2>🟢 Low-Priority Issues ({len(low_issues)})</h2>',
            self._render_issues_list(low_issues, "low", detail_level),
            f'</div>',
            
            # Generate summary stats if available
            self._generate_summary_stats(evaluated_result),
            
            f'</div>'  # Close issues-report div
        ]
        
        return "\n".join(html)
    
    def _render_issues_list(self, issues: List[Dict[str, Any]], severity: str, detail_level: str) -> str:
        """
        Render a list of issues as HTML.
        
        Args:
            issues: List of issue dictionaries
            severity: Severity level (critical, high, medium, low)
            detail_level: Level of detail to include
            
        Returns:
            HTML string for the issues list
        """
        if not issues:
            return "<p>No issues found in this category.</p>"
        
        html_parts = []
        
        for issue in issues:
            # Extract issue details with fallbacks
            title = issue.get("title", "Untitled Issue")
            description = issue.get("description", "")
            impact = issue.get("impact", "")
            context = issue.get("context", "")
            category = issue.get("category", "")
            
            # Build issue card
            issue_html = [
                f'<div class="issue-card {severity}">',
                f'<h3>{title}</h3>',
                
                f'<div class="issue-meta">',
                f'<span class="badge badge-{severity}">{severity.capitalize()}</span>',
            ]
            
            # Add category if available
            if category:
                issue_html.append(f'<span class="category-badge">{category}</span>')
            
            issue_html.append('</div>')  # Close issue-meta
            
            # Issue content section
            issue_html.append('<div class="issue-content">')
            issue_html.append(f'<p><strong>Description:</strong> {description}</p>')
            
            # Include additional details based on detail level
            if detail_level != "essential":
                if impact:
                    issue_html.append(f'<p><strong>Impact:</strong> {impact}</p>')
                
                if detail_level == "comprehensive" and context:
                    issue_html.append(f'<p><strong>Context:</strong> {context}</p>')
            
            issue_html.append('</div>')  # Close issue-content
            issue_html.append('</div>')  # Close issue-card
            
            html_parts.append("\n".join(issue_html))
        
        return "\n".join(html_parts)
    
    def _format_actions_report(self, evaluated_result: Dict[str, Any], context) -> str:
        """
        Format actions data into a Streamlit-friendly HTML report.
        
        Args:
            evaluated_result: Evaluated actions data
            context: Processing context
            
        Returns:
            HTML report string
        """
        # Extract actions by priority
        immediate_actions = evaluated_result.get("immediate_actions", [])
        short_term_actions = evaluated_result.get("short_term_actions", [])
        long_term_actions = evaluated_result.get("long_term_actions", [])
        
        # Get executive summary if available
        executive_summary = evaluated_result.get("executive_summary", 
                                                "No executive summary available.")
        
        # Build the HTML report
        html = [
            f'<div class="actions-report">',
            f'<h1>Action Items Report</h1>',
            
            # Executive Summary section
            f'<div class="executive-summary">',
            f'<h2>📋 Executive Summary</h2>',
            f'<p>{executive_summary}</p>',
            f'</div>',
            
            # Immediate Actions section
            f'<div class="actions-section">',
            f'<h2>🔴 Immediate Actions ({len(immediate_actions)})</h2>',
            self._render_actions_list(immediate_actions, "immediate"),
            f'</div>',
            
            # Short-term Actions section
            f'<div class="actions-section">',
            f'<h2>🟠 Short-Term Actions ({len(short_term_actions)})</h2>',
            self._render_actions_list(short_term_actions, "short-term"),
            f'</div>',
            
            # Long-term Actions section
            f'<div class="actions-section">',
            f'<h2>🟢 Long-Term Actions ({len(long_term_actions)})</h2>',
            self._render_actions_list(long_term_actions, "long-term"),
            f'</div>',
            
            f'</div>'  # Close actions-report div
        ]
        
        return "\n".join(html)
    
    def _render_actions_list(self, actions: List[Dict[str, Any]], priority: str) -> str:
        """
        Render a list of actions as HTML.
        
        Args:
            actions: List of action dictionaries
            priority: Priority level (immediate, short-term, long-term)
            
        Returns:
            HTML string for the actions list
        """
        if not actions:
            return "<p>No actions found in this category.</p>"
        
        html_parts = []
        
        for action in actions:
            # Extract action details with fallbacks
            title = action.get("title", "Untitled Action")
            description = action.get("description", "")
            owner = action.get("owner", "Unassigned")
            due_date = action.get("due_date", "")
            
            # Build action card
            action_html = [
                f'<div class="action-card {priority}">',
                f'<h3>{title}</h3>',
                
                f'<div class="action-meta">',
                f'<span class="action-owner">{owner}</span>',
            ]
            
            # Add due date if available
            if due_date:
                action_html.append(f'<span class="action-due-date">{due_date}</span>')
            
            action_html.append('</div>')  # Close action-meta
            
            # Action content section
            action_html.append('<div class="action-content">')
            action_html.append(f'<p>{description}</p>')
            action_html.append('</div>')  # Close action-content
            
            action_html.append('</div>')  # Close action-card
            
            html_parts.append("\n".join(action_html))
        
        return "\n".join(html_parts)
    
    def _format_generic_report(self, evaluated_result: Dict[str, Any], context) -> str:
        """
        Format generic data into a Streamlit-friendly HTML report.
        
        Args:
            evaluated_result: Evaluated data
            context: Processing context
            
        Returns:
            HTML report string
        """
        # Get executive summary if available
        executive_summary = evaluated_result.get("executive_summary", 
                                                "No executive summary available.")
        
        # Get any findings or items
        items = evaluated_result.get("findings", [])
        if not items:
            items = evaluated_result.get("items", [])
            if not items:
                # Try to find any list field
                for key, value in evaluated_result.items():
                    if isinstance(value, list) and len(value) > 0:
                        items = value
                        break
        
        # Build the HTML report
        html = [
            f'<div class="analysis-report">',
            f'<h1>{self.crew_type.capitalize()} Analysis Report</h1>',
            
            # Executive Summary section
            f'<div class="executive-summary">',
            f'<h2>📋 Executive Summary</h2>',
            f'<p>{executive_summary}</p>',
            f'</div>',
            
            # Findings section if available
            f'<div class="findings-section">',
            f'<h2>🔍 Key Findings</h2>',
            self._render_generic_items(items),
            f'</div>',
            
            f'</div>'  # Close analysis-report div
        ]
        
        return "\n".join(html)
    
    def _render_generic_items(self, items: List[Dict[str, Any]]) -> str:
        """
        Render a list of generic items as HTML.
        
        Args:
            items: List of item dictionaries
            
        Returns:
            HTML string for the items list
        """
        if not items:
            return "<p>No findings available.</p>"
        
        html_parts = []
        
        for item in items:
            if isinstance(item, dict):
                # Extract item details with fallbacks
                title = item.get("title", "")
                if not title:
                    title = item.get("name", "Untitled Item")
                
                description = item.get("description", "")
                if not description:
                    description = item.get("content", "")
                
                # Build item card
                item_html = [
                    f'<div class="item-card">',
                    f'<h3>{title}</h3>',
                    f'<div class="item-content">',
                    f'<p>{description}</p>',
                    f'</div>',
                    f'</div>'
                ]
                
                html_parts.append("\n".join(item_html))
            elif isinstance(item, str):
                # Simple string item
                html_parts.append(f'<div class="item-card"><p>{item}</p></div>')
        
        return "\n".join(html_parts)
    
    def _generate_summary_stats(self, evaluated_result: Dict[str, Any]) -> str:
        """
        Generate summary statistics section if available.
        
        Args:
            evaluated_result: Evaluated data
            
        Returns:
            HTML string for summary stats section
        """
        stats = evaluated_result.get("statistics", {})
        if not stats:
            stats = evaluated_result.get("summary_stats", {})
            if not stats:
                return ""
        
        html = [
            f'<div class="summary-stats">',
            f'<h2>📊 Summary Statistics</h2>',
            f'<ul>'
        ]
        
        for key, value in stats.items():
            formatted_key = key.replace("_", " ").title()
            html.append(f'<li><strong>{formatted_key}:</strong> {value}</li>')
        
        html.append('</ul>')
        html.append('</div>')
        
        return "\n".join(html)
    
    def _create_error_report(self, message: str) -> str:
        """
        Create an error report when formatting fails.
        
        Args:
            message: Error message
            
        Returns:
            HTML error report
        """
        html = [
            '<div class="issues-report">',
            '<h1>Issues Analysis Report</h1>',
            
            '<div class="error-message" style="background-color: rgba(255, 82, 82, 0.1); border-left: 4px solid #ff5252; padding: 15px; border-radius: 4px; margin: 20px 0;">',
            f'<h3>⚠️ Error During Report Generation</h3>',
            f'<p>{message}</p>',
            '<p>Please try again or check the document for potential issues.</p>',
            '</div>',
            
            '</div>'
        ]
        
        return "\n".join(html)
    
    async def format_report(
        self, 
        evaluated_items: Dict[str, Any], 
        document_info: Optional[Dict[str, Any]] = None, 
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Union[str, Dict[str, Any]]:
        """
        Legacy method for backward compatibility.
        Formats evaluated items into a structured report.
        
        Args:
            evaluated_items: Items to include in the report
            document_info: Optional document metadata
            user_preferences: Optional user formatting preferences
            
        Returns:
            Formatted report (HTML string or dictionary)
        """
        # Create a minimal context
        class MinimalContext:
            def __init__(self):
                self.results = {}
                self.document_info = {}
                self.options = {}
        
        context = MinimalContext()
        context.results["evaluation"] = evaluated_items
        context.document_info = document_info or {}
        context.options = user_preferences or {}
        
        # Use the new formatting methods
        if self.crew_type == "issues":
            return self._format_issues_report(evaluated_items, context)
        elif self.crew_type == "actions":
            return self._format_actions_report(evaluated_items, context)
        else:
            return self._format_generic_report(evaluated_items, context)
    
    def _sanitize_html_content(self, content: str) -> str:
        """
        Sanitize HTML content for Streamlit compatibility.
        
        Args:
            content: HTML content to sanitize
            
        Returns:
            Sanitized HTML content
        """
        # Remove script tags entirely (Streamlit blocks these)
        content = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', content)
        
        # Replace problematic characters
        content = content.replace('&nbsp;', ' ')
        
        # Ensure proper encoding of special characters
        content = content.replace('&', '&amp;')
        content = content.replace('<', '&lt;').replace('>', '&gt;')
        content = content.replace('&lt;div', '<div').replace('&lt;/div&gt;', '</div>')
        content = content.replace('&lt;h', '<h').replace('&lt;/h', '</h')
        content = content.replace('&lt;p', '<p').replace('&lt;/p&gt;', '</p>')
        content = content.replace('&lt;span', '<span').replace('&lt;/span&gt;', '</span>')
        content = content.replace('&lt;strong', '<strong').replace('&lt;/strong&gt;', '</strong>')
        content = content.replace('&lt;ul', '<ul').replace('&lt;/ul&gt;', '</ul>')
        content = content.replace('&lt;li', '<li').replace('&lt;/li&gt;', '</li>')
        
        # Ensure we have valid closing tags
        unclosed_divs = content.count('<div') - content.count('</div>')
        if unclosed_divs > 0:
            content += '</div>' * unclosed_divs
        
        return content
    
    def _get_stage_specific_content(self, context) -> str:
        """
        Get stage-specific content for the prompt.
        
        Args:
            context: Processing context
            
        Returns:
            Stage-specific content string
        """
        # Handle multiple context types
        if hasattr(context, 'results'):
            # Normal processing context
            evaluated_result = context.results.get("evaluation", {})
            detail_level = context.options.get("detail_level", "standard") if hasattr(context, "options") else "standard"
        elif isinstance(context, dict):
            # Dictionary context
            evaluated_result = context.get("evaluated_items", {})
            detail_level = context.get("detail_level", "standard")
        else:
            # Unknown context type
            return ""
        
        # Build content based on crew type
        content_parts = []
        
        # Add crew type guidance
        content_parts.append(f"REPORT TYPE: {self.crew_type}")
        
        # Add detail level guidance
        detail_guidance = self._get_detail_level_guidance(detail_level)
        content_parts.append(f"DETAIL LEVEL ({detail_level}): {detail_guidance}")
        
        # Add crew-specific guidance
        if self.crew_type == "issues":
            content_parts.append(self._get_issues_guidance())
        elif self.crew_type == "actions":
            content_parts.append(self._get_actions_guidance())
        
        return "\n\n".join(content_parts)
    
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
    
    def _get_issues_guidance(self) -> str:
        """
        Get guidance specific to issues reports.
        
        Returns:
            Issues guidance string
        """
        return """
ISSUES REPORT STRUCTURE:
1. Executive Summary - Overall assessment of issues found
2. Critical Issues - Problems requiring immediate attention
3. High-Priority Issues - Significant problems that need addressing
4. Medium-Priority Issues - Moderate problems to be fixed
5. Low-Priority Issues - Minor issues that should be noted

FORMATTING REQUIREMENTS:
- Use severity-appropriate styling (critical=red, high=orange, medium=yellow, low=green)
- Include badges for severity levels
- For each issue, include title, description, and impact
- Group similar issues together
- Ensure consistent formatting across all severity levels
"""
    
    def _get_actions_guidance(self) -> str:
        """
        Get guidance specific to actions reports.
        
        Returns:
            Actions guidance string
        """
        return """
ACTIONS REPORT STRUCTURE:
1. Executive Summary - Overall assessment of actions identified
2. Immediate Actions - Tasks requiring immediate attention
3. Short-Term Actions - Tasks to be completed in the near future
4. Long-Term Actions - Tasks with longer timelines

FORMATTING REQUIREMENTS:
- Highlight action owners and deadlines
- For each action, include title, description, owner, and timeline
- Group related actions together
- Ensure actions are clear and actionable
- Add visual priority indicators
"""