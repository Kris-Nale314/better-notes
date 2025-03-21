"""
UI enhancement utilities for Better Notes application.
Provides styling and visual improvement functions for the Streamlit interface.
"""

import streamlit as st

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
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .rich-output-container h3 {
            margin-top: 20px;
            color: rgba(255, 255, 255, 0.9);
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
    Enhance the markdown result text with styling and visual elements.
    
    Args:
        result_text: Original markdown result text
        analysis_type: Type of analysis ("issues", "actions", "insights")
        
    Returns:
        HTML string with enhanced markdown
    """
    # First enhance the markdown with icons
    enhanced_markdown = enhance_markdown_with_icons(result_text, analysis_type)
    
    # Then apply other enhancements like badges
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
    
    # Return the enhanced markdown wrapped in the rich output container
    return f"<div class='rich-output-container'>{enhanced_markdown}</div>"

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