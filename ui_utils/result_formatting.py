"""
Result formatting module for Better Notes.
Provides utilities for formatting and enhancing analysis output.
Updated to properly handle different content types including dictionaries.
"""

import streamlit as st
import re
import json
from typing import Dict, Any, Optional, List, Union

def enhance_result_display(result_text: Union[str, Dict, Any], analysis_type: str, detail_level: str = "standard") -> str:
    """
    Enhance the result text with styling and visual elements.
    Handles string, dictionary, or other content types.
    
    Args:
        result_text: Original result (string, dict, or other object)
        analysis_type: Type of analysis ("issues", "actions", "insights")
        detail_level: Level of detail ("essential", "standard", "comprehensive")
        
    Returns:
        HTML string with enhanced display
    """
    # Convert dictionary to string if needed
    if isinstance(result_text, dict):
        try:
            # Try to convert to JSON
            result_str = json.dumps(result_text, indent=2)
            # Wrap in pre tags for proper formatting
            return f'<div class="rich-output-container"><pre>{result_str}</pre></div>'
        except:
            # If JSON conversion fails, use string representation
            result_str = str(result_text)
    elif not isinstance(result_text, str):
        # Handle any non-string, non-dict objects
        result_str = str(result_text)
    else:
        result_str = result_text
    
    # Check if this is already HTML
    if is_html_content(result_str):
        return enhance_html_content(result_str, analysis_type)
    else:
        # Process as markdown
        return enhance_markdown_content(result_str, analysis_type)

def is_html_content(content: Any) -> bool:
    """
    Determine if content is HTML with improved error handling.
    
    Args:
        content: Content to check
        
    Returns:
        Boolean indicating if content is HTML
    """
    # Handle non-string content
    if not isinstance(content, str):
        return False
        
    html_indicators = [
        "<html", "<div", "<h1", "<p>", "<table", 
        "class=", "<body", "<head", "<script", "<style"
    ]
    
    try:
        # Check if any HTML indicators are present
        return any(indicator in content.lower() for indicator in html_indicators)
    except AttributeError:
        return False

def enhance_html_content(html_content: str, analysis_type: str) -> str:
    """
    Enhance HTML content with additional styling.
    
    Args:
        html_content: HTML content to enhance
        analysis_type: Type of analysis
        
    Returns:
        Enhanced HTML content
    """
    # Clean up any script tags for security
    cleaned_html = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', html_content)
    
    # Add wrapper if not already present
    if not ('<div class="rich-output-container"' in cleaned_html or 
            '<div class="issues-report"' in cleaned_html):
        
        container_class = "issues-report" if analysis_type == "issues" else "rich-output-container"
        cleaned_html = f'<div class="{container_class}">{cleaned_html}</div>'
    
    return cleaned_html

def enhance_markdown_content(markdown_text: str, analysis_type: str) -> str:
    """
    Enhance markdown content with icons and visual elements.
    
    Args:
        markdown_text: Markdown content to enhance
        analysis_type: Type of analysis
        
    Returns:
        Enhanced HTML from markdown
    """
    # Check if the text is valid for processing
    if not markdown_text or not isinstance(markdown_text, str):
        return f'<div class="rich-output-container">{str(markdown_text)}</div>'
    
    # Add icons to headings
    enhanced_text = add_icons_to_headings(markdown_text, analysis_type)
    
    # Apply specific formatting based on analysis type
    if analysis_type == "issues":
        enhanced_text = format_issues(enhanced_text)
    elif analysis_type == "actions":
        enhanced_text = format_actions(enhanced_text)
    elif analysis_type == "insights":
        enhanced_text = format_insights(enhanced_text)
    
    # Ensure proper newlines before headings for markdown rendering
    enhanced_text = re.sub(r'([^\n])(#+\s+)', r'\1\n\n\2', enhanced_text)
    
    # Return the enhanced markdown wrapped in the rich output container
    return f'<div class="rich-output-container">{enhanced_text}</div>'

def add_icons_to_headings(markdown_text: str, analysis_type: str) -> str:
    """
    Add icons to markdown headings.
    
    Args:
        markdown_text: Markdown text to enhance
        analysis_type: Type of analysis
        
    Returns:
        Markdown text with icons added to headings
    """
    # Common replacements for all analysis types
    replacements = {
        "# Executive Summary": "# 📋 Executive Summary",
        "# Summary": "# 📋 Summary",
        "## Summary": "## 📋 Summary",
        "# Introduction": "# 🚀 Introduction",
        "## Introduction": "## 🚀 Introduction",
        "# Conclusion": "# 🏁 Conclusion",
        "## Conclusion": "## 🏁 Conclusion",
        "# Recommendations": "# 💡 Recommendations",
        "## Recommendations": "## 💡 Recommendations",
        "# Key Findings": "# 🔑 Key Findings",
        "## Key Findings": "## 🔑 Key Findings",
    }
    
    # Analysis-specific replacements
    if analysis_type == "issues":
        replacements.update({
            "# Issues Identified": "# 🚨 Issues Identified",
            "## Critical Issues": "## 🔴 Critical Issues",
            "## High-Priority Issues": "## 🟠 High-Priority Issues",
            "## Medium-Priority Issues": "## 🟡 Medium-Priority Issues",
            "## Low-Priority Issues": "## 🟢 Low-Priority Issues",
            "## Summary Statistics": "## 📊 Summary Statistics",
        })
    elif analysis_type == "actions":
        replacements.update({
            "# Action Items Report": "# ✅ Action Items Report",
            "## Action Items by Owner": "## 👤 Action Items by Owner",
            "## Unassigned Action Items": "## ⚠️ Unassigned Action Items",
            "## Action Items by Timeframe": "## 📅 Action Items by Timeframe",
            "## Summary Table": "## 📊 Summary Table",
        })
    elif analysis_type == "insights":
        replacements.update({
            "# Document Insights Report": "# 💡 Document Insights Report",
            "## Document Overview": "## 📄 Document Overview",
            "## Key Themes": "## 🔑 Key Themes",
            "## Notable Quotes": "## 💬 Notable Quotes",
            "## Interesting Observations": "## 🔎 Interesting Observations",
            "## Context Cloud": "## ☁️ Context Cloud",
        })
    
    # Apply all replacements
    enhanced_text = markdown_text
    for original, replacement in replacements.items():
        enhanced_text = enhanced_text.replace(original, replacement)
    
    # Handle raw emoji in headings (already in the source)
    enhanced_text = re.sub(r'(##\s+[🔴🟠🟡🟢⚠️📊])', r'\1 ', enhanced_text)
    
    # Add horizontal rules between sections for better visual separation
    lines = enhanced_text.split('\n')
    for i in range(1, len(lines)):
        if lines[i].startswith('## ') and not lines[i-1].strip() == '---':
            lines.insert(i, '---')
    
    enhanced_text = '\n'.join(lines)
    
    return enhanced_text

def format_issues(markdown_text: str) -> str:
    """
    Format issues with severity badges and enhance layout.
    
    Args:
        markdown_text: Markdown text to format
        
    Returns:
        Formatted markdown text
    """
    # Add badges for severity levels
    formatted_text = markdown_text.replace(
        "**Severity:** critical", 
        "**Severity:** critical <span class='badge badge-critical'>Critical</span>"
    )
    formatted_text = formatted_text.replace(
        "**Severity:** high", 
        "**Severity:** high <span class='badge badge-high'>High</span>"
    )
    formatted_text = formatted_text.replace(
        "**Severity:** medium", 
        "**Severity:** medium <span class='badge badge-medium'>Medium</span>"
    )
    formatted_text = formatted_text.replace(
        "**Severity:** low", 
        "**Severity:** low <span class='badge badge-low'>Low</span>"
    )
    
    # Fix issue headers that might have emojis without proper spacing
    formatted_text = re.sub(r'(## 🔴) Critical Issues', r'## 🔴 Critical Issues', formatted_text)
    formatted_text = re.sub(r'(## 🟠) High-Priority Issues', r'## 🟠 High-Priority Issues', formatted_text)
    formatted_text = re.sub(r'(## 🟡) Medium-Priority Issues', r'## 🟡 Medium-Priority Issues', formatted_text)
    formatted_text = re.sub(r'(## 🟢) Low-Priority Issues', r'## 🟢 Low-Priority Issues', formatted_text)
    
    # Wrap executive summary in a special div if it exists
    if "# 📋 Executive Summary" in formatted_text:
        parts = formatted_text.split("# 📋 Executive Summary", 1)
        exec_part = parts[1].split("#", 1)
        if len(exec_part) > 1:
            formatted_text = (
                parts[0] + 
                "# 📋 Executive Summary" + 
                f"<div class='executive-summary'>{exec_part[0]}</div>#" + 
                exec_part[1]
            )
    
    return formatted_text

def format_actions(markdown_text: str) -> str:
    """
    Format actions with assignee and date highlighting.
    
    Args:
        markdown_text: Markdown text to format
        
    Returns:
        Formatted markdown text
    """
    # Highlight owner assignments
    formatted_text = re.sub(
        r'\*\*Assigned to:\*\* ([^*\n]+)', 
        r'**Assigned to:** <span class="action-owner">\1</span>', 
        markdown_text
    )
    
    # Highlight due dates
    formatted_text = re.sub(
        r'\*\*Due date:\*\* ([^*\n]+)', 
        r'**Due date:** <span class="action-due-date">\1</span>', 
        formatted_text
    )
    
    # Wrap executive summary in a special div if it exists
    if "# 📋 Executive Summary" in formatted_text:
        parts = formatted_text.split("# 📋 Executive Summary", 1)
        exec_part = parts[1].split("#", 1)
        if len(exec_part) > 1:
            formatted_text = (
                parts[0] + 
                "# 📋 Executive Summary" + 
                f"<div class='executive-summary'>{exec_part[0]}</div>#" + 
                exec_part[1]
            )
    
    return formatted_text

def format_insights(markdown_text: str) -> str:
    """
    Format insights with tags and quote highlighting.
    
    Args:
        markdown_text: Markdown text to format
        
    Returns:
        Formatted markdown text
    """
    # Format quotes with special styling
    formatted_text = re.sub(
        r'> (.+?)(?:\n\n|\n$)', 
        r'<div class="quote-box">\1</div>\n\n', 
        markdown_text
    )
    
    # Add tags formatting
    formatted_text = re.sub(
        r'#([a-zA-Z0-9_]+)', 
        r'<span class="insight-tag">#\1</span>', 
        formatted_text
    )
    
    # Wrap executive summary in a special div if it exists
    if "# 📋 Executive Summary" in formatted_text:
        parts = formatted_text.split("# 📋 Executive Summary", 1)
        exec_part = parts[1].split("#", 1)
        if len(exec_part) > 1:
            formatted_text = (
                parts[0] + 
                "# 📋 Executive Summary" + 
                f"<div class='executive-summary'>{exec_part[0]}</div>#" + 
                exec_part[1]
            )
    
    return formatted_text

def format_log_entries(log_entries: List[str]) -> str:
    """
    Format log entries for display.
    
    Args:
        log_entries: List of log entry strings
        
    Returns:
        HTML string with formatted log entries
    """
    log_html = []
    for entry in log_entries:
        # Add highlighting for different log types
        entry_class = "log-entry"
        if "error" in entry.lower() or "failed" in entry.lower():
            entry_class += " status-error"
        elif "success" in entry.lower() or "complete" in entry.lower():
            entry_class += " status-complete"
        elif "working" in entry.lower() or "processing" in entry.lower():
            entry_class += " status-working"
            
        log_html.append(f'<div class="{entry_class}">{entry}</div>')
    
    return "\n".join(log_html)

def create_download_button(content: Any, filename: str, mime_type: str = "text/html"):
    """
    Create a styled download button for content with improved handling of content types.
    
    Args:
        content: Content to download (string, dict, or other object)
        filename: File name for download
        mime_type: MIME type of content
    """
    # Convert content to string if not already
    if isinstance(content, dict):
        try:
            content_str = json.dumps(content, indent=2)
            # Use plain text mime type for JSON
            mime_type = "text/plain"
        except:
            content_str = str(content)
    elif not isinstance(content, str):
        content_str = str(content)
        mime_type = "text/plain"
    else:
        content_str = content
    
    st.download_button(
        "📥 Download Report",
        data=content_str,
        file_name=filename,
        mime=mime_type,
        key=f"download_{filename}",
        help="Download the analysis report to your device"
    )

def extract_html_content(result: Any) -> str:
    """
    Extract HTML content from different result formats.
    
    Args:
        result: Result object (could be string, dict, or other)
        
    Returns:
        HTML string or empty string if no HTML found
    """
    # Handle string directly
    if isinstance(result, str):
        if is_html_content(result):
            return result
        return ""
    
    # Handle dictionary
    if isinstance(result, dict):
        # Look for HTML content in any string field
        for key, value in result.items():
            if isinstance(value, str) and is_html_content(value):
                return value
                
        # Check for specific keys
        if "raw_output" in result and isinstance(result["raw_output"], str):
            return result["raw_output"]
        elif "formatted_result" in result and isinstance(result["formatted_result"], str):
            return result["formatted_result"]
        elif "content" in result and isinstance(result["content"], str):
            return result["content"]
            
    # If no HTML found, return empty string
    return ""