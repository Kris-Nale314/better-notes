"""
Better Notes - AI-powered document analysis and note-taking assistant.
Main entry point for the Streamlit application.
"""

import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import UI utilities
from ui_utils.core_styling import apply_component_styles, apply_analysis_styles
from lean.options import ProcessingOptions

# Check for OpenAI API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    import sys
    print("⚠️  WARNING: OPENAI_API_KEY environment variable not found.")
    print("Please set your API key in a .env file or environment variable.")

# Create necessary directories
for directory in ["data", "outputs", ".cache", "agents/config", "ui_utils"]:
    Path(directory).mkdir(exist_ok=True, parents=True)

# Configure Streamlit page
st.set_page_config(
    page_title="Better Notes",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
apply_component_styles()

# Additional custom CSS for more vibrant UI elements
st.markdown("""
<style>
    /* Purple title for better-notes */
    .purple-title {
        color: #6c5ce7;
        font-weight: 700;
        font-size: 2.5rem;
        letter-spacing: -1px;
    }
    
    /* White subtitle */
    .white-subtitle {
        color: white;
        font-weight: 400;
    }
    
    /* Vibrant accent colors */
    .accent-red {
        color: #ff5252;
        font-weight: 600;
    }
    
    .accent-yellow {
        color: #ffca28;
        font-weight: 600;
    }
    
    .accent-blue {
        color: #4e8cff;
        font-weight: 600;
    }
    
    .accent-green {
        color: #20c997;
        font-weight: 600;
    }
    
    .accent-purple {
        color: #6c5ce7;
        font-weight: 600;
    }
    
    .accent-orange {
        color: #ff9f43;
        font-weight: 600;
    }
    
    /* Enhanced feature cards */
    .feature-card {
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
        border-left: 5px solid;
        background-color: rgba(50, 50, 70, 0.2);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
    }
    
    .summary-card {
        border-color: #4e8cff;
        box-shadow: 0 4px 10px rgba(78, 140, 255, 0.1);
    }
    
    .actions-card {
        border-color: #20c997;
        box-shadow: 0 4px 10px rgba(32, 201, 151, 0.1);
    }
    
    .issues-card {
        border-color: #ff5252;
        box-shadow: 0 4px 10px rgba(255, 82, 82, 0.1);
    }
    
    .insights-card {
        border-color: #ffca28;
        box-shadow: 0 4px 10px rgba(255, 202, 40, 0.1);
    }
    
    /* Enhanced flow diagram */
    .flow-diagram {
        background: linear-gradient(135deg, rgba(50, 50, 70, 0.2), rgba(70, 70, 90, 0.3));
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    
    .flow-step {
        width: 120px;
        padding: 10px;
        margin: 0 auto 15px auto;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        font-weight: 500;
        color: white;
    }
    
    .flow-arrow {
        height: 20px;
        margin-left: 60px;
        color: rgba(255, 255, 255, 0.5);
    }
    
    /* Colorful number highlights */
    .number-red {
        background-color: #ff5252;
        color: white;
        border-radius: 50%;
        width: 36px;
        height: 36px;
        display: inline-flex;
        justify-content: center;
        align-items: center;
        font-weight: bold;
        margin-right: 10px;
    }
    
    .number-yellow {
        background-color: #ffca28;
        color: #333;
        border-radius: 50%;
        width: 36px;
        height: 36px;
        display: inline-flex;
        justify-content: center;
        align-items: center;
        font-weight: bold;
        margin-right: 10px;
    }
    
    .number-blue {
        background-color: #4e8cff;
        color: white;
        border-radius: 50%;
        width: 36px;
        height: 36px;
        display: inline-flex;
        justify-content: center;
        align-items: center;
        font-weight: bold;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Main Page Content ---
st.markdown('<h1 style="color: #a29bfe; font-weight: 700; font-size: 2.5rem; letter-spacing: -1px;">better-notes</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="color: white; font-weight: 400;">Transform Documents Into Organized, Insightful Notes</h3>', unsafe_allow_html=True)

# Introduction
st.markdown("""
Better Notes is an AI-powered document analysis tool that goes beyond basic summarization.
Our <span class='accent-purple'>multi-agent approach</span> breaks down complex documents into structured insights with deeper understanding than traditional AI.
""", unsafe_allow_html=True)

# Key Benefits section in columns with red, yellow, blue colors
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div>
        <span class="number-red">1</span> <b class="accent-red">Specialized Agents</b>
        <p>A team of AI agents work together to analyze your documents, each focusing on specific aspects for deeper insights.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div>
        <span class="number-yellow">2</span> <b class="accent-yellow">Structured Analysis</b>
        <p>Information is categorized, prioritized, and organized for quick understanding and action.</p>
    </div>
    """, unsafe_allow_html=True)
    
with col3:
    st.markdown("""
    <div>
        <span class="number-blue">3</span> <b class="accent-blue">Interactive Exploration</b>
        <p>Chat with your document, refine analysis, and extract exactly what you need.</p>
    </div>
    """, unsafe_allow_html=True)

# About Agent-Based Analysis Section
with st.expander("How Agent-Based Analysis Works", expanded=False):
    agent_col1, agent_col2 = st.columns([3, 2])
    
    with agent_col1:
        # Add top padding to align with the flow diagram's header
        st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### The Power of <span class="accent-purple">Collaborative AI</span>
        """, unsafe_allow_html=True)
        
        # Simplified agent description with less vertical space
        st.markdown("""
        <p style="margin-bottom: 8px;">Better Notes uses specialized AI agents that collaborate like a team of analysts:</p>
        
        <div style="font-size: 0.9rem; line-height: 1.4;">
        <p style="margin: 3px 0;"><span class="accent-orange">Planner Agent</span> creates a document-specific analysis strategy</p>
        <p style="margin: 3px 0;"><span class="accent-blue">Extractor Agents</span> identify valuable information from document chunks</p>
        <p style="margin: 3px 0;"><span class="accent-green">Aggregator Agent</span> combines and deduplicates findings</p>
        <p style="margin: 3px 0;"><span class="accent-purple">Evaluator Agent</span> assesses importance and relevance</p>
        <p style="margin: 3px 0;"><span class="accent-orange">Formatter Agent</span> creates a clear, structured final report</p>
        <p style="margin: 3px 0;"><span class="accent-red">Reviewer Agent</span> performs quality control on the final output</p>
        </div>
        
        <p style="margin-top: 10px;">This collaborative approach produces higher quality results than single-model analysis by leveraging specialization and focused expertise.</p>
        """, unsafe_allow_html=True)
    
    with agent_col2:
        # Container with centered content - compact styling
        st.write('<div class="flow-diagram" style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 10px;">', unsafe_allow_html=True)
        st.write('<h4 style="text-align: center; width: 100%; margin: 0 0 8px 0; font-size: 0.9rem;">Agent Collaboration Flow</h4>', unsafe_allow_html=True)
        
        # Document emoji - smaller and less spacing
        st.write('<div style="font-size: 30px; margin-bottom: 2px; text-align: center; width: 100%;">📄</div>', unsafe_allow_html=True)
        st.write('<div style="height: 6px; width: 2px; background-color: rgba(255,255,255,0.5); margin: 0 auto 2px auto;"></div>', unsafe_allow_html=True)
        
        # Planner step - narrower and less margin
        st.write('<div style="background-color: #ff9f43; margin: 0 auto 2px auto; text-align: center; width: 90px; padding: 3px 5px; border-radius: 4px; font-size: 0.8rem; color: white;">Planner</div>', unsafe_allow_html=True)
        st.write('<div style="height: 6px; width: 2px; background-color: rgba(255,255,255,0.5); margin: 0 auto 2px auto;"></div>', unsafe_allow_html=True)
        
        # Extractors step - compact
        st.write('<div style="background-color: #4e8cff; margin: 0 auto 2px auto; text-align: center; width: 90px; padding: 3px 5px; border-radius: 4px; font-size: 0.8rem; color: white;">Extractors</div>', unsafe_allow_html=True)
        st.write('<div style="height: 6px; width: 2px; background-color: rgba(255,255,255,0.5); margin: 0 auto 2px auto;"></div>', unsafe_allow_html=True)
        
        # Aggregator step - compact
        st.write('<div style="background-color: #20c997; margin: 0 auto 2px auto; text-align: center; width: 90px; padding: 3px 5px; border-radius: 4px; font-size: 0.8rem; color: white;">Aggregator</div>', unsafe_allow_html=True)
        st.write('<div style="height: 6px; width: 2px; background-color: rgba(255,255,255,0.5); margin: 0 auto 2px auto;"></div>', unsafe_allow_html=True)
        
        # Evaluator step - compact
        st.write('<div style="background-color: #6c5ce7; margin: 0 auto 2px auto; text-align: center; width: 90px; padding: 3px 5px; border-radius: 4px; font-size: 0.8rem; color: white;">Evaluator</div>', unsafe_allow_html=True)
        st.write('<div style="height: 6px; width: 2px; background-color: rgba(255,255,255,0.5); margin: 0 auto 2px auto;"></div>', unsafe_allow_html=True)
        
        # Formatter step - compact
        st.write('<div style="background-color: #ff9f43; margin: 0 auto 2px auto; text-align: center; width: 90px; padding: 3px 5px; border-radius: 4px; font-size: 0.8rem; color: white;">Formatter</div>', unsafe_allow_html=True)
        st.write('<div style="height: 6px; width: 2px; background-color: rgba(255,255,255,0.5); margin: 0 auto 2px auto;"></div>', unsafe_allow_html=True)
        
        # Reviewer step - compact
        st.write('<div style="background-color: #ff5252; margin: 0 auto 2px auto; text-align: center; width: 90px; padding: 3px 5px; border-radius: 4px; font-size: 0.8rem; color: white;">Reviewer</div>', unsafe_allow_html=True)
        st.write('</div>', unsafe_allow_html=True)

# Feature cards in columns with color adjustments
st.subheader("Choose Your Analysis Type")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card summary-card">
        <h3>📝 Document Summarization</h3>
        <p>Transform lengthy documents into concise, organized summaries with 
        adjustable detail levels. Works well with meeting transcripts, research
        papers, reports, and articles.</p>
        <p><a href="/Summary">Open Summarization →</a></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-card actions-card">
        <h3>✅ Action Item Extraction</h3>
        <p>Identify tasks, assignments, commitments, and follow-up
        items to ensure nothing falls through the cracks.</p>
        <p><a href="/Multi_Agent">Open Action Items →</a></p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card issues-card">
        <h3>🔍 Issue Identification</h3>
        <p>Extract problems, challenges, and areas of concern 
        from your documents to focus improvement efforts.</p>
        <p><a href="/Assess_Issues">Open Issue Identification →</a></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card insights-card">
        <h3>💡 Context Insights</h3>
        <p>Discover key themes, tone, participants, and notable statements to
        quickly understand what matters in your document.</p>
        <p><a href="/Multi_Agent">Open Context Insights →</a></p>
    </div>
    """, unsafe_allow_html=True)

# Architecture visualization with enhanced styling
st.markdown("""
<div style="background: linear-gradient(135deg, rgba(50, 50, 70, 0.3), rgba(70, 70, 90, 0.4)); 
            border-radius: 12px; 
            padding: 25px; 
            margin-top: 30px;">
    <h2 style="text-align: center; margin-bottom: 20px;">Powered by <span class="accent-purple">Multi-Agent Architecture</span></h2>
    <div align="center">
        <img src="https://raw.githubusercontent.com/kris-nale314/better-notes/main/docs/images/logic.svg" alt="Better-Notes Logic" width="90%"/>
        <p style="margin-top: 15px; font-style: italic;">Each assessment type uses a team of specialized agents coordinated by a Planner that creates document-specific instructions</p>
    </div>
</div>
""", unsafe_allow_html=True)

# App information in expander with accent colors
with st.expander("About Better Notes"):
    st.markdown("""
    ## How It Works
    
    Better Notes uses a sophisticated architecture with these key components:
    
    1. <span class="accent-blue">Smart Chunking</span>: Divides documents into meaningful macro-chunks (10k tokens each)
    2. <span class="accent-orange">Multi-agent Processing</span>: Assigns specialized analysis tasks to expert agents
    3. <span class="accent-green">Collaborative Synthesis</span>: Combines insights into a coherent whole
    4. <span class="accent-purple">Post-Analysis Interaction</span>: Chat with your document and refine analysis
    
    The application is built with a modular design that makes it easy to extend
    with new analysis types through the crew system.
    
    ## Getting Started
    
    1. Select an analysis type from the options above
    2. Upload a document (text files work best)
    3. Adjust processing options as needed
    4. Generate your enhanced analysis
    5. Chat with your document to explore further
    
    For best results, ensure you're using a clean text document. Meeting
    transcripts, reports, articles, and research papers work particularly well.
    """, unsafe_allow_html=True)

# Check for API key with styled warning
if not api_key:
    st.markdown("""
    <div style="background-color: rgba(255, 82, 82, 0.2); 
                border-left: 5px solid #ff5252; 
                padding: 15px; 
                border-radius: 5px;">
        ⚠️ <b>OpenAI API key not found!</b> Please set the OPENAI_API_KEY environment variable 
        before using the application.
    </div>
    """, unsafe_allow_html=True)

# Display version with styled footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px; background: rgba(108, 92, 231, 0.1); border-radius: 8px; border: 1px solid rgba(108, 92, 231, 0.2);">
    <span style="font-size: 0.8em; color: #6c5ce7;">better-notes v0.1.4</span>
</div>
""", unsafe_allow_html=True)