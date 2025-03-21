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

from lean.options import ProcessingOptions
from ui_utils.ui_enhance import apply_custom_css


# Check for OpenAI API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    import sys
    print("⚠️  WARNING: OPENAI_API_KEY environment variable not found.")
    print("Please set your API key in a .env file or environment variable.")

# Create necessary directories
for directory in ["data", "outputs", ".cache", "agents/config"]:
    Path(directory).mkdir(exist_ok=True, parents=True)

# Configure Streamlit page
st.set_page_config(
    page_title="Better Notes",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()


# --- Main Page Content ---
st.title("Better Notes")
st.markdown("### Transform Documents Into Organized, Insightful Notes")

# Introduction
st.markdown("""
Better Notes is an AI-powered document analysis tool that goes beyond basic summarization.
Our multi-agent approach breaks down complex documents into structured insights with deeper understanding than traditional AI.
""")

# Key Benefits section in columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div>
        <span class="big-number">1.</span> <b>Deeper Analysis</b>
        <p>Specialized AI agents focus on specific aspects of your documents, extracting more meaningful insights.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div>
        <span class="big-number">2.</span> <b>Better Organization</b>
        <p>Information is categorized, prioritized, and structured for quick understanding and action.</p>
    </div>
    """, unsafe_allow_html=True)
    
with col3:
    st.markdown("""
    <div>
        <span class="big-number">3.</span> <b>Customizable Output</b>
        <p>Control the detail level, focus areas, and presentation format to match your needs.</p>
    </div>
    """, unsafe_allow_html=True)

# About Agent-Based Analysis Section
with st.expander("How Agent-Based Analysis Works", expanded=False):
    agent_col1, agent_col2 = st.columns([3, 2])
    
    with agent_col1:
        st.markdown("""
        ### The Power of Collaborative AI
        
        Better Notes uses specialized AI agents that collaborate like a team of analysts:
        
        1. **Extractor Agents** analyze document chunks to identify specific elements
        2. **Aggregator Agent** combines and deduplicates findings
        3. **Evaluator Agent** assesses importance and relevance
        4. **Formatter Agent** creates a clear, structured final report
        
        This collaborative approach produces higher quality results than single-model analysis by leveraging specialization and focused expertise, just like human teams outperform individuals.
        """)
    
    with agent_col2:
        # Simple diagram showing agent collaboration with HTML
        st.markdown("""
        <div class="flow-diagram">
            <h4>Agent Collaboration Flow</h4>
            <div style="margin: 10px auto; width: 180px;">
                <div class="flow-box" style="background-color: #4e8cff;">Document</div>
                <div style="border-left: 2px dashed #888; height: 15px; margin-left: 90px;"></div>
                <div class="flow-box" style="background-color: #6c5ce7;">Extractors</div>
                <div style="border-left: 2px solid #888; height: 15px; margin-left: 90px;"></div>
                <div class="flow-box" style="background-color: #00b894;">Aggregator</div>
                <div style="border-left: 2px solid #888; height: 15px; margin-left: 90px;"></div>
                <div class="flow-box" style="background-color: #fdcb6e;">Evaluator</div>
                <div style="border-left: 2px solid #888; height: 15px; margin-left: 90px;"></div>
                <div class="flow-box" style="background-color: #e17055;">Formatter</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Feature cards in columns
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
        
        <p><a href="/Multi_Agent">Open Issue Identification →</a></p>
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

# App information in expander
with st.expander("About Better Notes"):
    st.markdown("""
    ## How It Works
    
    Better Notes uses a sophisticated architecture with these key components:
    
    1. **Smart Chunking**: Divides documents into meaningful sections
    2. **Multi-agent Processing**: Assigns specialized analysis tasks to expert agents
    3. **Collaborative Synthesis**: Combines insights into a coherent whole
    4. **Customizable Output**: Formats results based on your preferences
    
    The application is built with a modular design that makes it easy to extend
    with new analysis types through the crew system.
    
    ## Getting Started
    
    1. Select an analysis type from the options above
    2. Upload a document (text files work best)
    3. Adjust processing options as needed
    4. Generate your enhanced analysis
    
    For best results, ensure you're using a clean text document. Meeting
    transcripts, reports, articles, and research papers work particularly well.
    """)

# Check for API key
if not api_key:
    st.warning(
        "⚠️ OpenAI API key not found! Please set the OPENAI_API_KEY environment variable "
        "before using the application."
    )

# Display version
st.sidebar.markdown("---")
st.sidebar.caption("Better Notes v0.1.2")