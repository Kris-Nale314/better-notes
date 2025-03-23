"""
Core styling module for Better Notes UI components.
Provides essential styling for the application with modular approach.
"""

import streamlit as st

def apply_core_styles():
    """
    Apply essential base styling for the application.
    Only includes core elements needed across all pages.
    """
    st.markdown("""
    <style>
        /* === BASE STYLING === */
        /* Improved spacing and typography */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        h1, h2, h3 {
            margin-top: 1rem;
            margin-bottom: 1rem;
            font-weight: 600;
        }
        
        p {
            margin-bottom: 1rem;
            line-height: 1.5;
        }
        
        /* === CARD ELEMENTS === */
        /* Base card styling used by multiple components */
        .card {
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            background-color: rgba(70, 70, 70, 0.2);
            border-left: 4px solid #4e8cff;
        }
        
        /* Status colors for cards */
        .status-waiting {
            border-left-color: #6c757d;
            background-color: rgba(108, 117, 125, 0.2);
        }
        
        .status-working {
            border-left-color: #ff9f43;
            background-color: rgba(255, 159, 67, 0.2);
        }
        
        .status-complete {
            border-left-color: #20c997;
            background-color: rgba(32, 201, 151, 0.2);
        }
        
        .status-error {
            border-left-color: #ff5252;
            background-color: rgba(255, 82, 82, 0.2);
        }
        
        /* === CONTENT DISPLAY === */
        /* Output containers */
        .output-container {
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            background-color: rgba(50, 50, 60, 0.2);
            border-left: 5px solid #4e8cff;
        }
        
        /* Section dividers */
        .section-divider {
            margin: 2rem 0;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Alert/info boxes */
        .info-box {
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            background-color: rgba(108, 92, 231, 0.1);
            border-left: 4px solid #6c5ce7;
        }
        
        /* === SEVERITY INDICATORS === */
        /* Severity badges and indicators */
        .badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 500;
            margin-left: 0.5rem;
        }
        
        .badge-critical {
            background-color: rgba(255, 82, 82, 0.2);
            color: #ff5252;
            border: 1px solid rgba(255, 82, 82, 0.3);
        }
        
        .badge-high {
            background-color: rgba(255, 159, 67, 0.2);
            color: #ff9f43;
            border: 1px solid rgba(255, 159, 67, 0.3);
        }
        
        .badge-medium {
            background-color: rgba(253, 203, 110, 0.2);
            color: #fdcb6e;
            border: 1px solid rgba(253, 203, 110, 0.3);
        }
        
        .badge-low {
            background-color: rgba(32, 201, 151, 0.2);
            color: #20c997;
            border: 1px solid rgba(32, 201, 151, 0.3);
        }
        
        /* Dot indicators for compact displays */
        .dot-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        
        .dot-critical { background-color: #ff5252; }
        .dot-high { background-color: #ff9f43; }
        .dot-medium { background-color: #fdcb6e; }
        .dot-low { background-color: #20c997; }
        
        /* === STRUCTURAL ELEMENTS === */
        /* Flow diagram elements */
        .flow-diagram {
            background-color: rgba(60, 60, 80, 0.1);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        
        .flow-step {
            padding: 8px;
            text-align: center;
            border-radius: 5px;
            margin-bottom: 15px;
            color: white;
            font-weight: 500;
        }
        
        /* Feature highlighting */
        .big-number {
            font-size: 28px;
            font-weight: bold;
            margin-right: 10px;
        }
        
        /* === TABLE STYLING === */
        /* Enhanced tables */
        .styled-table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }
        
        .styled-table th {
            background-color: rgba(60, 60, 70, 0.4);
            padding: 0.75rem;
            text-align: left;
            font-weight: 600;
        }
        
        .styled-table td {
            padding: 0.75rem;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .styled-table tr:nth-child(even) {
            background-color: rgba(60, 60, 70, 0.2);
        }
    </style>
    """, unsafe_allow_html=True)

def apply_component_styles():
    """
    Apply styling for specific components beyond core styles.
    Call this when you need the complete set of styles.
    """
    # First apply core styles
    apply_core_styles()
    
    # Then add component-specific styles
    st.markdown("""
    <style>
        /* === AGENT CARDS === */
        .agent-card {
            display: flex;
            flex-direction: column;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            background-color: rgba(70, 70, 70, 0.2);
            border-left: 4px solid #4e8cff;
            transition: transform 0.2s;
        }
        
        .agent-card:hover {
            transform: translateY(-2px);
        }
        
        .agent-card.working {
            border-left-color: #ff9f43;
            background-color: rgba(255, 159, 67, 0.1);
        }
        
        .agent-card.complete {
            border-left-color: #20c997;
            background-color: rgba(32, 201, 151, 0.1);
        }
        
        .agent-card.error {
            border-left-color: #ff5252;
            background-color: rgba(255, 82, 82, 0.1);
        }
        
        .agent-card h4 {
            margin: 0 0 10px 0;
        }
        
        .agent-card p {
            margin: 5px 0;
            font-size: 14px;
        }
        
        /* === ISSUES REPORT STYLING === */
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
        }
        
        .executive-summary {
            background-color: rgba(108, 92, 231, 0.1);
            border-radius: 8px;
            padding: 15px 20px;
            margin: 20px 0;
            border-left: 4px solid #6c5ce7;
        }
        
        .issue-card {
            background-color: rgba(60, 60, 70, 0.3);
            border-radius: 8px;
            margin: 15px 0;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            border-left: 4px solid;
        }
        
        .issue-card h3 {
            font-size: 1.15em;
            margin: 0;
            padding: 12px 15px;
            background-color: rgba(30, 30, 40, 0.4);
        }
        
        .issue-meta {
            padding: 5px 15px;
            background-color: rgba(30, 30, 40, 0.2);
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .issue-content {
            padding: 15px;
        }
        
        .issue-card.critical { border-left-color: #ff5252; }
        .issue-card.high { border-left-color: #ff9f43; }
        .issue-card.medium { border-left-color: #fdcb6e; }
        .issue-card.low { border-left-color: #20c997; }
        
        /* === CHAT INTERFACE === */
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
        
        /* === PROGRESS INDICATORS === */
        .step-progress-container {
            display: flex;
            justify-content: space-between;
            margin: 20px 0;
            position: relative;
        }
        
        .step-progress-bar {
            position: absolute;
            top: 15px;
            left: 0;
            height: 2px;
            background-color: rgba(255, 255, 255, 0.1);
            width: 100%;
            z-index: 1;
        }
        
        .step-progress-fill {
            height: 100%;
            background-color: #6c5ce7;
            transition: width 0.3s ease;
        }
        
        .step-item {
            z-index: 2;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        
        .step-icon {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background-color: rgba(70, 70, 70, 0.3);
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 8px;
            transition: all 0.3s;
        }
        
        .step-icon.active {
            background-color: #6c5ce7;
            transform: scale(1.1);
            box-shadow: 0 0 10px rgba(108, 92, 231, 0.5);
        }
        
        .step-icon.complete {
            background-color: #20c997;
        }
        
        .step-label {
            font-size: 12px;
            font-weight: 500;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

def apply_analysis_styles(analysis_type=None):
    """
    Apply styling specific to a particular analysis type.
    
    Args:
        analysis_type: Type of analysis ("issues", "actions", "insights")
    """
    # Apply core styles first
    apply_core_styles()
    
    # Common analysis styles
    st.markdown("""
    <style>
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
            border-left: 4px solid #ff9f43;
        }
        .actions-card {
            border-left: 4px solid #20c997;
        }
        .insights-card {
            border-left: 4px solid #6c5ce7;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Add analysis-specific styles if requested
    if analysis_type == "issues":
        st.markdown("""
        <style>
            /* Issues-specific styling */
            .issue-section-header {
                display: flex;
                align-items: center;
                margin-bottom: 15px;
            }
            
            .issue-section-header .count-badge {
                margin-left: 10px;
                background-color: rgba(108, 92, 231, 0.2);
                padding: 3px 10px;
                border-radius: 12px;
                font-size: 0.9em;
            }
            
            .issue-priority {
                font-weight: 600;
                display: inline-block;
                width: 80px;
            }
        </style>
        """, unsafe_allow_html=True)
    elif analysis_type == "actions":
        st.markdown("""
        <style>
            /* Actions-specific styling */
            .action-owner {
                background-color: rgba(32, 201, 151, 0.1);
                padding: 3px 8px;
                border-radius: 4px;
                font-size: 0.9em;
                margin-right: 8px;
            }
            
            .action-due-date {
                background-color: rgba(108, 92, 231, 0.1);
                padding: 3px 8px;
                border-radius: 4px;
                font-size: 0.9em;
            }
        </style>
        """, unsafe_allow_html=True)
    elif analysis_type == "insights":
        st.markdown("""
        <style>
            /* Insights-specific styling */
            .insight-tag {
                background-color: rgba(108, 92, 231, 0.1);
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                margin-right: 5px;
                display: inline-block;
            }
            
            .quote-box {
                background-color: rgba(70, 70, 90, 0.2);
                border-left: 4px solid #6c5ce7;
                padding: 12px 15px;
                margin: 15px 0;
                font-style: italic;
                border-radius: 0 8px 8px 0;
            }
        </style>
        """, unsafe_allow_html=True)

def highlight_theme():
    """
    Apply a highlight on top of the Streamlit theme for special elements.
    Can be used for temporary emphasis.
    """
    st.markdown("""
    <style>
        .highlight-container {
            background-color: rgba(108, 92, 231, 0.15);
            border-radius: 8px;
            padding: 20px;
            margin: 25px 0;
            border: 1px solid rgba(108, 92, 231, 0.3);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% {
                box-shadow: 0 0 0 0 rgba(108, 92, 231, 0.4);
            }
            70% {
                box-shadow: 0 0 0 8px rgba(108, 92, 231, 0);
            }
            100% {
                box-shadow: 0 0 0 0 rgba(108, 92, 231, 0);
            }
        }
    </style>
    """, unsafe_allow_html=True)