"""
Updated FormatterAgent that produces HTML compatible with Streamlit's dark theme.
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
    Formatter agent that creates structured output for Streamlit with dark theme compatibility.
    Creates well-formatted reports that look good on both light and dark themes.
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
        
        logger.info(f"FormatterAgent initialized for {crew_type}")
    
    async def process(self, context):
        """
        Format evaluation results into a structured report.
        
        Args:
            context: ProcessingContext object
            
        Returns:
            Formatted report as HTML
        """
        logger.info("FormatterAgent starting formatting process")
        
        try:
            # Get evaluation results from context
            evaluated_result = context.results.get("evaluation", {})
            
            if not evaluated_result:
                logger.warning("No evaluation results found for formatting")
                return self._create_error_report("No evaluation results available")
            
            # Format the report based on crew type
            if self.crew_type == "issues":
                formatted_result = self._format_issues_report(evaluated_result, context)
            else:
                formatted_result = self._format_generic_report(evaluated_result, context)
            
            logger.info("Successfully formatted report")
            return formatted_result
            
        except Exception as e:
            logger.error(f"Error in formatting process: {e}")
            
            # Return error report
            return self._create_error_report(f"Error formatting report: {str(e)}")
    
    def _format_issues_report(self, evaluated_result: Dict[str, Any], context) -> str:
        """
        Format issues into a clean HTML report for Streamlit with dark theme compatibility.
        
        Args:
            evaluated_result: Evaluated issues
            context: ProcessingContext
            
        Returns:
            HTML report
        """
        # Get issues by severity
        critical_issues = self._get_issues_by_severity(evaluated_result, "critical")
        high_issues = self._get_issues_by_severity(evaluated_result, "high")
        medium_issues = self._get_issues_by_severity(evaluated_result, "medium")
        low_issues = self._get_issues_by_severity(evaluated_result, "low")
        
        # Get executive summary if available, or generate one
        executive_summary = evaluated_result.get("executive_summary", "")
        if not executive_summary:
            executive_summary = self._generate_summary(
                critical_issues, high_issues, medium_issues, low_issues
            )
        
        # Get user preferences
        user_preferences = context.options if hasattr(context, "options") else {}
        detail_level = user_preferences.get("detail_level", "standard")
        
        # Add CSS for dark theme compatibility
        css = self._get_dark_theme_css()
        
        # Build the HTML report
        html = [
            f'<style>{css}</style>',
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
            
            f'</div>'  # Close issues-report div
        ]
        
        return "\n".join(html)
    
    def _get_dark_theme_css(self) -> str:
        """
        Get CSS for dark theme compatibility.
        
        Returns:
            CSS string
        """
        return """
        /* Base Styles */
        .issues-report {
            color: rgba(250, 250, 250, 0.95);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            padding: 1rem;
            line-height: 1.5;
        }
        
        .issues-report h1, .issues-report h2, .issues-report h3 {
            color: rgba(255, 255, 255, 0.95);
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        
        .issues-report p {
            color: rgba(220, 220, 220, 0.95);
            margin-bottom: 1rem;
        }
        
        /* Card Styles */
        .issue-card {
            background-color: rgba(32, 33, 36, 0.6);
            border-radius: 6px;
            padding: 1rem;
            margin-bottom: 1rem;
            border-left: 4px solid #666;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
        }
        
        .issue-card.critical {
            border-left-color: #ff5252;
        }
        
        .issue-card.high {
            border-left-color: #ff9f43;
        }
        
        .issue-card.medium {
            border-left-color: #ffc107;
        }
        
        .issue-card.low {
            border-left-color: #20c997;
        }
        
        .issue-card h3 {
            margin-top: 0;
            color: rgba(255, 255, 255, 0.95);
        }
        
        /* Meta Info Styles */
        .issue-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 0.75rem;
        }
        
        .badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            font-size: 0.75rem;
            font-weight: 600;
            border-radius: 999px;
            background-color: #666;
            color: #fff;
        }
        
        .badge-critical {
            background-color: #ff5252;
        }
        
        .badge-high {
            background-color: #ff9f43;
        }
        
        .badge-medium {
            background-color: #ffc107;
            color: #333;
        }
        
        .badge-low {
            background-color: #20c997;
            color: #333;
        }
        
        .category-badge {
            background-color: rgba(102, 126, 234, 0.6);
            color: white;
            padding: 0.25rem 0.5rem;
            font-size: 0.75rem;
            border-radius: 999px;
        }
        
        /* Content Styles */
        .issue-content {
            margin-top: 0.75rem;
        }
        
        .issue-content p {
            margin-bottom: 0.5rem;
        }
        
        .issue-content strong {
            font-weight: 600;
            color: rgba(255, 255, 255, 0.9);
        }
        
        /* Section Styles */
        .executive-summary {
            background-color: rgba(44, 49, 60, 0.5);
            border-radius: 6px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            border-left: 4px solid #4e8cff;
        }
        
        .issues-section {
            margin-bottom: 2rem;
        }
        
        /* Error Message */
        .error-message {
            background-color: rgba(255, 82, 82, 0.1);
            border-left: 4px solid #ff5252;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
            color: rgba(250, 250, 250, 0.95);
        }
        
        /* Source Information */
        .source-info {
            font-size: 0.8rem;
            color: rgba(200, 200, 200, 0.8);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            padding-top: 0.5rem;
            margin-top: 0.5rem;
        }
        """
    
    def _get_issues_by_severity(self, evaluated_result: Dict[str, Any], severity: str) -> List[Dict[str, Any]]:
        """
        Extract issues of a specific severity from evaluation results.
        
        Args:
            evaluated_result: Evaluated results
            severity: Severity level to extract
            
        Returns:
            List of issues with the specified severity
        """
        # Check for direct severity lists in the result
        severity_key = f"{severity}_issues"
        if severity_key in evaluated_result and isinstance(evaluated_result[severity_key], list):
            return evaluated_result[severity_key]
        
        # Otherwise, look in evaluated_issues
        issues = []
        if "evaluated_issues" in evaluated_result and isinstance(evaluated_result["evaluated_issues"], list):
            for issue in evaluated_result["evaluated_issues"]:
                if isinstance(issue, dict) and issue.get("severity", "") == severity:
                    issues.append(issue)
        
        return issues
    
    def _generate_summary(
        self, 
        critical_issues: List[Dict[str, Any]], 
        high_issues: List[Dict[str, Any]],
        medium_issues: List[Dict[str, Any]], 
        low_issues: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a simple summary of issues found.
        
        Args:
            critical_issues: List of critical issues
            high_issues: List of high issues
            medium_issues: List of medium issues
            low_issues: List of low issues
            
        Returns:
            Generated summary
        """
        total_issues = len(critical_issues) + len(high_issues) + len(medium_issues) + len(low_issues)
        
        if total_issues == 0:
            return "No issues were identified in this document."
        
        summary = f"Analysis identified {total_issues} total issues: "
        summary += f"{len(critical_issues)} critical, {len(high_issues)} high, "
        summary += f"{len(medium_issues)} medium, and {len(low_issues)} low priority. "
        
        if len(critical_issues) > 0:
            summary += "Critical attention is needed for " + self._list_issue_titles(critical_issues, 3) + ". "
            
        if len(high_issues) > 0:
            summary += "High priority issues include " + self._list_issue_titles(high_issues, 3) + "."
            
        return summary
    
    def _list_issue_titles(self, issues: List[Dict[str, Any]], max_issues: int = 3) -> str:
        """
        Create a comma-separated list of issue titles.
        
        Args:
            issues: List of issues
            max_issues: Maximum number of issues to list
            
        Returns:
            Comma-separated list of titles
        """
        if not issues:
            return ""
            
        titles = [issue.get("title", "Untitled Issue") for issue in issues[:max_issues]]
        
        if len(issues) > max_issues:
            return ", ".join(titles) + f", and {len(issues) - max_issues} more"
        else:
            return ", ".join(titles)
    
    def _render_issues_list(self, issues: List[Dict[str, Any]], severity: str, detail_level: str) -> str:
        """
        Render a list of issues as HTML.
        
        Args:
            issues: List of issue dictionaries
            severity: Severity level
            detail_level: Detail level (essential, standard, comprehensive)
            
        Returns:
            HTML for the issues list
        """
        if not issues:
            return "<p>No issues found in this category.</p>"
        
        html_parts = []
        
        for issue in issues:
            # Extract issue details with fallbacks
            title = issue.get("title", "Untitled Issue")
            description = issue.get("description", "")
            impact = issue.get("impact", issue.get("impact_assessment", ""))
            category = issue.get("category", "")
            
            # Build issue card
            issue_html = [
                f'<div class="issue-card {severity}">',
                f'<h3>{title}</h3>',
                
                f'<div class="issue-meta">',
                f'<span class="badge badge-{severity}">{severity.capitalize()}</span>'
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
                
                if detail_level == "comprehensive":
                    # Add source information for comprehensive view
                    if "source_chunks" in issue:
                        chunks = issue["source_chunks"]
                        if chunks and isinstance(chunks, list):
                            issue_html.append(f'<p class="source-info"><strong>Mentioned in:</strong> {len(chunks)} document sections</p>')
            
            issue_html.append('</div>')  # Close issue-content
            issue_html.append('</div>')  # Close issue-card
            
            html_parts.append("\n".join(issue_html))
        
        return "\n".join(html_parts)
    
    def _format_generic_report(self, evaluated_result: Dict[str, Any], context) -> str:
        """
        Format generic results into a clean HTML report compatible with dark theme.
        
        Args:
            evaluated_result: Evaluated results
            context: ProcessingContext
            
        Returns:
            HTML report
        """
        # Get executive summary if available
        executive_summary = evaluated_result.get("executive_summary", 
                                                "No executive summary available.")
        
        # Get items
        items = []
        if "findings" in evaluated_result and isinstance(evaluated_result["findings"], list):
            items = evaluated_result["findings"]
        elif "items" in evaluated_result and isinstance(evaluated_result["items"], list):
            items = evaluated_result["items"]
        elif "evaluated_items" in evaluated_result and isinstance(evaluated_result["evaluated_items"], list):
            items = evaluated_result["evaluated_items"]
        
        # Add CSS for dark theme
        css = self._get_dark_theme_css()
        
        # Build the HTML report
        html = [
            f'<style>{css}</style>',
            f'<div class="analysis-report">',
            f'<h1>{self.crew_type.capitalize()} Analysis Report</h1>',
            
            # Executive Summary section
            f'<div class="executive-summary">',
            f'<h2>📋 Executive Summary</h2>',
            f'<p>{executive_summary}</p>',
            f'</div>',
            
            # Items section
            f'<div class="items-section">',
            f'<h2>🔍 Findings</h2>',
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
            HTML for the items list
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
                    f'<div class="issue-card">',
                    f'<h3>{title}</h3>',
                    f'<div class="issue-content">',
                    f'<p>{description}</p>',
                    f'</div>',
                    f'</div>'
                ]
                
                html_parts.append("\n".join(item_html))
            elif isinstance(item, str):
                # Simple string item
                html_parts.append(f'<div class="issue-card"><p>{item}</p></div>')
        
        return "\n".join(html_parts)
    
    def _create_error_report(self, message: str) -> str:
        """
        Create an error report when formatting fails, compatible with dark theme.
        
        Args:
            message: Error message
            
        Returns:
            HTML error report
        """
        # Add CSS for dark theme
        css = self._get_dark_theme_css()
        
        html = [
            f'<style>{css}</style>',
            '<div class="issues-report">',
            '<h1>Analysis Report</h1>',
            
            '<div class="error-message">',
            f'<h3>⚠️ Error During Report Generation</h3>',
            f'<p>{message}</p>',
            '<p>Please try again or check the document for potential issues.</p>',
            '</div>',
            
            '</div>'
        ]
        
        return "\n".join(html)