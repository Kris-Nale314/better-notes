"""
UI enhancement utilities for Better Notes application.
Provides styling and visual improvement functions for the Streamlit interface.
Includes chat interface components.
"""

import streamlit as st
import asyncio
import time
from typing import List, Dict, Any, Callable, Optional

def apply_custom_css():
    """
    Apply custom CSS styling to the Streamlit interface.
    Should be called near the top of each page.
    """
    st.markdown("""
    <style>
        /* Agent cards styling */
        .agent-card {
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0px;
            background-color: rgba(70, 70, 70, 0.2);
            border-left: 4px solid #4e8cff;
        }
        .agent-working {
            border-left: 4px solid #ff9f43;
            background-color: rgba(80, 70, 60, 0.2);
        }
        .agent-complete {
            border-left: 4px solid #20c997;
            background-color: rgba(60, 80, 60, 0.2);
        }
        
        /* Analysis type cards */
        .analysis-card {
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0px;
            transition: transform 0.2s, box-shadow 0.2s;
            background-color: rgba(60, 60, 70, 0.2);
        }
        .analysis-card:hover {
            transform: translateY(-2px);
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.2);
        }
        .issues-card {
            border-left: 4px solid #4e8cff;
        }
        .actions-card {
            border-left: 4px solid #ff9f43;
        }
        .insights-card {
            border-left: 4px solid #20c997;
        }
        
        /* Log entries */
        .log-entry {
            padding: 5px 10px;
            margin: 5px 0;
            border-radius: 4px;
            background-color: rgba(60, 60, 70, 0.3);
            font-family: monospace;
        }
        
        /* Feature cards */
        .feature-card {
            border-radius: 8px;
            padding: 18px;
            margin: 12px 0px;
            border-left: 4px solid;
            background-color: rgba(70, 70, 70, 0.1);
        }
        .summary-card {
            border-left-color: #4e8cff;
        }
        .issues-card {
            border-left-color: #ff9f43;
        }
        .actions-card {
            border-left-color: #20c997; 
        }
        .insights-card {
            border-left-color: #9b59b6;
        }
        
        /* Rich HTML Issues Report Styling */
        .issues-report {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: rgba(255, 255, 255, 0.9);
            padding: 0 10px;
        }
        
        .issues-report h1 {
            font-size: 1.8em;
            margin-bottom: 25px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
        }
        
        .issues-report h2 {
            font-size: 1.4em;
            margin-top: 30px;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            position: relative;
        }
        
        .issues-report .executive-summary {
            background-color: rgba(108, 92, 231, 0.1);
            border-radius: 8px;
            padding: 15px 20px;
            margin: 20px 0;
            border-left: 4px solid #6c5ce7;
        }
        
        .issues-report .issue-card {
            background-color: rgba(60, 60, 70, 0.3);
            border-radius: 8px;
            margin: 15px 0;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            border-left: 4px solid;
        }
        
        .issues-report .issue-card h3 {
            font-size: 1.15em;
            margin: 0;
            padding: 12px 15px;
            background-color: rgba(30, 30, 40, 0.4);
        }
        
        .issues-report .issue-meta {
            padding: 5px 15px;
            background-color: rgba(30, 30, 40, 0.2);
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .issues-report .issue-content {
            padding: 15px;
        }
        
        .issues-report .issue-content p {
            margin: 10px 0;
        }
        
        .issues-report .issue-card.critical {
            border-left-color: #ff5252;
        }
        
        .issues-report .issue-card.high {
            border-left-color: #ff9f43;
        }
        
        .issues-report .issue-card.medium {
            border-left-color: #fdcb6e;
        }
        
        .issues-report .issue-card.low {
            border-left-color: #20c997;
        }
        
        .issues-report .severity {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 500;
        }
        
        .issues-report .severity.critical {
            background-color: rgba(255, 80, 80, 0.2);
            color: #ff5252;
            border: 1px solid rgba(255, 80, 80, 0.3);
        }
        
        .issues-report .severity.high {
            background-color: rgba(255, 159, 67, 0.2);
            color: #ff9f43;
            border: 1px solid rgba(255, 159, 67, 0.3);
        }
        
        .issues-report .severity.medium {
            background-color: rgba(253, 203, 110, 0.2);
            color: #fdcb6e;
            border: 1px solid rgba(253, 203, 110, 0.3);
        }
        
        .issues-report .severity.low {
            background-color: rgba(32, 201, 151, 0.2);
            color: #20c997;
            border: 1px solid rgba(32, 201, 151, 0.3);
        }
        
        .issues-report .summary-stats table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        
        .issues-report .summary-stats th {
            background-color: rgba(60, 60, 70, 0.4);
            padding: 10px;
            text-align: left;
            font-weight: 600;
        }
        
        .issues-report .summary-stats td {
            padding: 10px;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .issues-report .summary-stats tr:nth-child(even) {
            background-color: rgba(60, 60, 70, 0.2);
        }
        
        .issues-report .summary-stats .total-row {
            border-top: 2px solid rgba(255, 255, 255, 0.1);
            background-color: rgba(60, 60, 70, 0.3);
        }
        
        .issues-report .severity-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 5px;
        }
        
        .issues-report .severity-indicator.critical {
            background-color: #ff5252;
        }
        
        .issues-report .severity-indicator.high {
            background-color: #ff9f43;
        }
        
        .issues-report .severity-indicator.medium {
            background-color: #fdcb6e;
        }
        
        .issues-report .severity-indicator.low {
            background-color: #20c997;
        }
        
        .issues-report .recommendations {
            background-color: rgba(70, 70, 90, 0.2);
            border-radius: 8px;
            padding: 15px 20px;
            margin: 25px 0;
        }
        
        .issues-report .recommendations ul {
            margin: 10px 0;
            padding-left: 25px;
        }
        
        .issues-report .recommendations li {
            margin-bottom: 10px;
        }

        /* Flow diagram */
        .flow-diagram {
            background-color: rgba(60, 60, 80, 0.1);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        .flow-box {
            padding: 8px;
            text-align: center;
            border-radius: 5px;
            margin-bottom: 15px;
            color: white;
            font-weight: 500;
        }
        
        /* Numbering */
        .big-number {
            font-size: 28px;
            font-weight: bold;
            margin-right: 10px;
        }
        
        /* Rich output styling */
        .rich-output-container {
            border-radius: 10px;
            padding: 10px;
            margin: 20px 0;
            background-color: rgba(50, 50, 60, 0.2);
            border-left: 5px solid #4e8cff;
        }
        
        .rich-output-container h1 {
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 20px;
        }
        
        .rich-output-container h2 {
            margin-top: 25px;
            padding-bottom: 8px;
            padding-top: 15px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            font-size: 1.5em;
            clear: both;
        }
        
        .rich-output-container h3 {
            margin-top: 20px;
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.3em;
        }
        
        /* Fix for emoji in headers */
        .rich-output-container h2 img.emojione,
        .rich-output-container h2 span.emoji {
            display: inline;
            vertical-align: middle;
            margin-right: 0.2em;
        }
        
        .rich-output-container blockquote {
            background-color: rgba(70, 70, 80, 0.3);
            border-left: 3px solid #6c5ce7;
            padding: 10px 15px;
            margin: 10px 0;
            border-radius: 0 5px 5px 0;
        }
        
        .rich-output-container table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        
        .rich-output-container th {
            background-color: rgba(80, 80, 100, 0.3);
            padding: 10px;
            text-align: left;
        }
        
        .rich-output-container td {
            padding: 8px 10px;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .rich-output-container tr:nth-child(even) {
            background-color: rgba(60, 60, 70, 0.2);
        }
        
        .rich-output-container code {
            background-color: rgba(40, 40, 50, 0.5);
            padding: 2px 5px;
            border-radius: 3px;
            font-family: monospace;
        }
        
        .rich-output-container ul, .rich-output-container ol {
            padding-left: 25px;
            margin: 10px 0;
        }
        
        .rich-output-container li {
            margin-bottom: 5px;
        }
        
        /* Severity badges */
        .critical-badge, .high-badge, .medium-badge, .low-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 500;
            margin-left: 8px;
        }
        
        .critical-badge {
            background-color: rgba(255, 80, 80, 0.2);
            color: #ff5252;
            border: 1px solid rgba(255, 80, 80, 0.3);
        }
        
        .high-badge {
            background-color: rgba(255, 159, 67, 0.2);
            color: #ff9f43;
            border: 1px solid rgba(255, 159, 67, 0.3);
        }
        
        .medium-badge {
            background-color: rgba(253, 203, 110, 0.2);
            color: #fdcb6e;
            border: 1px solid rgba(253, 203, 110, 0.3);
        }
        
        .low-badge {
            background-color: rgba(32, 201, 151, 0.2);
            color: #20c997;
            border: 1px solid rgba(32, 201, 151, 0.3);
        }
        
        /* Executive summary box */
        .executive-summary {
            background-color: rgba(108, 92, 231, 0.1);
            border-radius: 8px;
            padding: 15px;
            border-left: 4px solid #6c5ce7;
            margin: 20px 0;
        }
        
        /* Chat interface styling */
        .chat-container {
            border-radius: 10px;
            padding: 15px;
            margin: 20px 0;
            background-color: rgba(50, 50, 60, 0.2);
            border: 1px solid rgba(100, 100, 120, 0.3);
        }
        
        .chat-message {
            padding: 10px 15px;
            margin: 8px 0;
            border-radius: 8px;
            max-width: 85%;
        }
        
        .user-message {
            background-color: rgba(70, 130, 180, 0.2);
            border: 1px solid rgba(70, 130, 180, 0.3);
            margin-left: auto;
            margin-right: 10px;
            border-bottom-right-radius: 2px;
        }
        
        .assistant-message {
            background-color: rgba(60, 60, 70, 0.2);
            border: 1px solid rgba(60, 60, 70, 0.3);
            margin-right: auto;
            margin-left: 10px;
            border-bottom-left-radius: 2px;
        }
        
        .chat-input-container {
            padding: 10px;
            background-color: rgba(60, 60, 70, 0.1);
            border-radius: 8px;
            margin-top: 15px;
        }
        
        .quick-question-button {
            margin: 0 5px 10px 0;
            padding: 5px 10px;
            border-radius: 15px;
            background-color: rgba(108, 92, 231, 0.1);
            border: 1px solid rgba(108, 92, 231, 0.3);
            color: rgba(255, 255, 255, 0.9);
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .quick-question-button:hover {
            background-color: rgba(108, 92, 231, 0.2);
            transform: translateY(-1px);
        }
    </style>
    """, unsafe_allow_html=True)

def render_agent_card(agent_type, status, component):
    """
    Render an agent status card.
    
    Args:
        agent_type: Type of agent (extractor, aggregator, etc.)
        status: Status (waiting, working, complete)
        component: Streamlit component to render to
    """
    agent_info = {
        "extractor": {
            "title": "Extractor Agent", 
            "role": "Identifying patterns"
        },
        "aggregator": {
            "title": "Aggregator Agent", 
            "role": "Combining and deduplicating"
        },
        "evaluator": {
            "title": "Evaluator Agent", 
            "role": "Assessing importance and relevance"
        },
        "formatter": {
            "title": "Formatter Agent", 
            "role": "Creating structured report"
        }
    }
    
    status_icons = {
        "waiting": "⏳",
        "working": "🔄",
        "complete": "✅"
    }
    
    status_classes = {
        "waiting": "agent-card",
        "working": "agent-card agent-working",
        "complete": "agent-card agent-complete"
    }
    
    component.markdown(f"""
    <div class="{status_classes[status]}">
        <h4>{status_icons[status]} {agent_info[agent_type]['title']}</h4>
        <p>{agent_info[agent_type]['role']}</p>
        <p><small>Status: {status.capitalize()}</small></p>
    </div>
    """, unsafe_allow_html=True)

def enhance_markdown_with_icons(markdown_text, analysis_type):
    """
    Add icons and visual enhancements to the markdown output.
    
    Args:
        markdown_text: Original markdown text
        analysis_type: Type of analysis
        
    Returns:
        Enhanced markdown with icons and visual elements
    """
    # Add icons to headings
    enhanced_text = markdown_text
    
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
    for original, replacement in replacements.items():
        enhanced_text = enhanced_text.replace(original, replacement)
    
    # Also handle raw emoji in headings (already in the source)
    # This regex matches headings with emojis but without the proper space after
    import re
    enhanced_text = re.sub(r'(##\s+[🔴🟠🟡🟢⚠️📊])', r'\1 ', enhanced_text)
    
    # Add horizontal rules between sections for better visual separation
    # Look for level 2 headings (##) and add a horizontal rule before them if they don't already have one
    lines = enhanced_text.split('\n')
    for i in range(1, len(lines)):
        if lines[i].startswith('## ') and not lines[i-1].strip() == '---':
            lines.insert(i, '---')
    
    enhanced_text = '\n'.join(lines)
    
    return enhanced_text

def enhance_result_display(result_text, analysis_type):
    """
    Enhance the result text with styling and visual elements.
    
    Args:
        result_text: Original result text (markdown or HTML)
        analysis_type: Type of analysis ("issues", "actions", "insights")
        
    Returns:
        HTML string with enhanced display
    """
    # Import module explicitly to avoid any namespace issues
    import re as regex_module
    
    # Check if this is already HTML (issues now return HTML)
    if analysis_type == "issues":
        if "<div" in result_text or "<h1" in result_text:
            # Clean up any script tags for security
            cleaned_text = regex_module.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', result_text)
            # Directly return the HTML content
            return cleaned_text
    
    # Otherwise, process as markdown (for backward compatibility)
    # First enhance the markdown with icons
    enhanced_markdown = enhance_markdown_with_icons(result_text, analysis_type)
    
    # Then apply other enhancements
    if analysis_type == "issues":
        # Add badges for severity levels
        enhanced_markdown = enhanced_markdown.replace(
            "**Severity:** critical", 
            "**Severity:** critical <span class='critical-badge'>Critical</span>"
        )
        enhanced_markdown = enhanced_markdown.replace(
            "**Severity:** high", 
            "**Severity:** high <span class='high-badge'>High</span>"
        )
        enhanced_markdown = enhanced_markdown.replace(
            "**Severity:** medium", 
            "**Severity:** medium <span class='medium-badge'>Medium</span>"
        )
        enhanced_markdown = enhanced_markdown.replace(
            "**Severity:** low", 
            "**Severity:** low <span class='low-badge'>Low</span>"
        )
        
        # Fix issue headers that might have emojis without proper Markdown formatting
        enhanced_markdown = regex_module.sub(r'(## 🔴) Critical Issues', r'## 🔴 Critical Issues', enhanced_markdown)
        enhanced_markdown = regex_module.sub(r'(## 🟠) High-Priority Issues', r'## 🟠 High-Priority Issues', enhanced_markdown)
        enhanced_markdown = regex_module.sub(r'(## 🟡) Medium-Priority Issues', r'## 🟡 Medium-Priority Issues', enhanced_markdown)
        enhanced_markdown = regex_module.sub(r'(## 🟢) Low-Priority Issues', r'## 🟢 Low-Priority Issues', enhanced_markdown)
        
        # Process each issue into a better-formatted card
        enhanced_markdown = format_issues_as_cards(enhanced_markdown)
    
    # Wrap executive summary in a special div if it exists
    if "# 📋 Executive Summary" in enhanced_markdown:
        parts = enhanced_markdown.split("# 📋 Executive Summary", 1)
        exec_part = parts[1].split("#", 1)
        if len(exec_part) > 1:
            enhanced_markdown = (
                parts[0] + 
                "# 📋 Executive Summary" + 
                f"<div class='executive-summary'>{exec_part[0]}</div>#" + 
                exec_part[1]
            )
    
    # Ensure proper newlines before headings - critical for markdown rendering
    enhanced_markdown = regex_module.sub(r'([^\n])(#+\s+)', r'\1\n\n\2', enhanced_markdown)
    
    # Return the enhanced markdown wrapped in the rich output container
    return f"<div class='rich-output-container'>{enhanced_markdown}</div>"

def format_issues_as_cards(markdown_text):
    """
    Format issue entries as visually distinct cards.
    
    Args:
        markdown_text: The markdown text to process
        
    Returns:
        Processed markdown with issue cards
    """
    # Import module explicitly to avoid any namespace issues
    import re as regex_module
    
    # Split the markdown into sections for processing
    sections = []
    current_section = []
    current_severity = None
    
    lines = markdown_text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for main section headers (Critical, High, etc.)
        if regex_module.match(r'^## 🔴 Critical Issues', line):
            if current_section:
                sections.append((current_severity, current_section))
            current_section = [line]
            current_severity = "critical"
        elif regex_module.match(r'^## 🟠 High-Priority Issues', line):
            if current_section:
                sections.append((current_severity, current_section))
            current_section = [line]
            current_severity = "high"
        elif regex_module.match(r'^## 🟡 Medium-Priority Issues', line):
            if current_section:
                sections.append((current_severity, current_section))
            current_section = [line]
            current_severity = "medium"
        elif regex_module.match(r'^## 🟢 Low-Priority Issues', line):
            if current_section:
                sections.append((current_severity, current_section))
            current_section = [line]
            current_severity = "low"
        # Check for other main headers
        elif regex_module.match(r'^## ', line) or regex_module.match(r'^# ', line):
            if current_section:
                sections.append((current_severity, current_section))
            current_section = [line]
            current_severity = None
        else:
            current_section.append(line)
        i += 1
    
    # Add the last section
    if current_section:
        sections.append((current_severity, current_section))
    
    # Process each section
    result = []
    for severity, section in sections:
        if severity in ["critical", "high", "medium", "low"]:
            # Process issue entries in this section
            result.extend(process_issue_section(section, severity))
        else:
            # Keep section as is
            result.extend(section)
    
    return '\n'.join(result)


def process_issue_section(section_lines, severity):
    """
    Process a section of issues to format individual issues as cards.
    
    Args:
        section_lines: List of lines in the section
        severity: Severity level of this section
        
    Returns:
        List of processed lines
    """
    # Import re module explicitly to avoid namespace issues
    import re as regex_module
    
    # First line is the section header, keep it
    result = [section_lines[0], ""]
    
    # Process the rest of the section
    issue_blocks = []
    current_issue = []
    
    # Check if there are any issues in this section
    no_issues_found = False
    for line in section_lines[1:]:
        if "No critical issues were identified" in line or "No high-priority issues were identified" in line or \
           "No medium-priority issues were identified" in line or "No low-priority issues were identified" in line:
            no_issues_found = True
            result.append(line)
            break
    
    if no_issues_found:
        return result
    
    # Process each issue
    i = 1
    while i < len(section_lines):
        line = section_lines[i]
        
        # Check for issue title (level 3 header)
        if line.startswith("### "):
            if current_issue:
                issue_blocks.append(current_issue)
            current_issue = [line]
        elif current_issue:
            current_issue.append(line)
        i += 1
    
    # Add the last issue
    if current_issue:
        issue_blocks.append(current_issue)
    
    # Format each issue block
    for issue in issue_blocks:
        card_lines = []
        
        # Extract title from first line (### Title)
        title = issue[0][4:].strip() if issue and issue[0].startswith("### ") else "Issue"
        
        # Start issue card
        card_lines.append(f'<div class="issue-card issue-{severity}">')
        card_lines.append(f'<h3>{title}</h3>')
        
        # Add the rest of the issue content
        for line in issue[1:]:
            card_lines.append(line)
        
        # Close issue card
        card_lines.append('</div>')
        
        # Add to result
        result.extend(card_lines)
    
    return result

def format_log_entries(log_entries):
    """
    Format log entries for display.
    
    Args:
        log_entries: List of log entry strings
        
    Returns:
        HTML string with formatted log entries
    """
    log_html = []
    for entry in log_entries:
        log_html.append(f'<div class="log-entry">{entry}</div>')
    
    return "\n".join(log_html)

# ---- Chat Interface Components ----

def initialize_chat_state():
    """Initialize session state variables for chat interface."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""

def display_chat_interface(
    llm_client, 
    document_text: str,
    summary_text: str,
    document_info: Optional[Dict[str, Any]] = None
):
    """
    Display a chat interface for interacting with the document.
    
    Args:
        llm_client: LLM client for generating responses
        document_text: Original document text
        summary_text: Summary text generated from the document
        document_info: Optional document metadata
    """
    initialize_chat_state()
    
    st.divider()
    st.subheader("💬 Ask About This Document")
    
    # Chat message display container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.chat_history:
            role = message["role"]
            content = message["content"]
            
            # Choose styling based on role
            if role == "user":
                message_class = "chat-message user-message"
                prefix = "You: "
            else:
                message_class = "chat-message assistant-message"
                prefix = "Assistant: "
            
            st.markdown(f"""
                <div class="{message_class}">
                    <strong>{prefix}</strong>{content}
                </div>
            """, unsafe_allow_html=True)
    
    # Quick question buttons
    st.markdown("<div style='display: flex; flex-wrap: wrap;'>", unsafe_allow_html=True)
    
    quick_questions = [
        "Can you provide more details?",
        "What are the key points?",
        "Simplify this for me",
        "What action items were mentioned?"
    ]
    
    # Create columns for the buttons
    cols = st.columns(4)
    for i, question in enumerate(quick_questions):
        with cols[i]:
            if st.button(question, key=f"quick_{i}"):
                process_chat_question(llm_client, question, document_text, summary_text, document_info)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # User question input
    with st.form(key="chat_form", clear_on_submit=True):
        user_question = st.text_input(
            "Your question:",
            key="chat_input",
            placeholder="Ask a question about this document..."
        )
        submit_button = st.form_submit_button("Send")
        
        if submit_button and user_question:
            process_chat_question(llm_client, user_question, document_text, summary_text, document_info)

def process_chat_question(
    llm_client, 
    question: str, 
    document_text: str,
    summary_text: str,
    document_info: Optional[Dict[str, Any]] = None
):
    """
    Process a chat question and add to history.
    
    Args:
        llm_client: LLM client for generating responses
        question: User's question
        document_text: Original document text
        summary_text: Summary text generated from the document
        document_info: Optional document metadata
    """
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": question})
    
    # Create a placeholder for the assistant's response
    with st.spinner("Thinking..."):
        # Generate context
        truncated_doc = document_text[:3000]
        
        # Extract document type and metadata
        doc_type = "transcript" if document_info and document_info.get("is_meeting_transcript") else "document"
        
        # Create prompt
        prompt = f"""
        You are an AI assistant helping with document analysis and questions.
        
        DOCUMENT SUMMARY:
        {summary_text}
        
        DOCUMENT TYPE: {doc_type}
        
        DOCUMENT EXCERPT (beginning of document):
        {truncated_doc}
        
        USER QUESTION: {question}
        
        Please answer the question based on the document information provided.
        Focus on being helpful, concise, and accurate.
        If the information is not available in the context, say so.
        """
        
        # Get response from LLM
        try:
            response = asyncio.run(llm_client.generate_completion_async(prompt))
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Force a rerun to update the display
            st.rerun()
        except Exception as e:
            # Handle error gracefully
            error_message = f"Sorry, I encountered an error while processing your question: {str(e)}"
            st.session_state.chat_history.append({"role": "assistant", "content": error_message})
            st.rerun()