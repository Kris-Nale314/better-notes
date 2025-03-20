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

# Check for OpenAI API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    import sys
    print("⚠️  WARNING: OPENAI_API_KEY environment variable not found.")
    print("Please set your API key in a .env file or environment variable.")

# Create necessary directories
for directory in ["data", "outputs", ".cache"]:
    Path(directory).mkdir(exist_ok=True)

# Configure Streamlit page
st.set_page_config(
    page_title="Better Notes",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Main Page Content ---
st.title("Better Notes")
st.markdown("### Transform Documents Into Organized, Insightful Notes")
st.markdown("""
Better Notes is an AI-powered document analysis tool that helps you:

- **Extract Key Information** from lengthy documents and transcripts
- **Generate Well-Structured Summaries** with adjustable detail levels
- **Identify Action Items** and other important elements
- **Save Time** processing meeting notes, research, and other text

Use the sidebar navigation to access different features:
""")

# Feature cards in columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📝 Document Summarization
    
    Transform lengthy documents into concise, organized summaries with 
    adjustable detail levels. Works well with meeting transcripts, research
    papers, reports, and articles.
    
    [Open Summarization](/Summary)
    """)

with col2:
    st.markdown("""
    ### 🔬 Specialized Analysis
    
    Extract specific information types from your documents 
    including action items, issues, and opportunities.
    
    *Coming soon...*
    """)

# App information in expander
with st.expander("About Better Notes"):
    st.markdown("""
    ## How It Works
    
    Better Notes uses a lean architecture with these key components:
    
    1. **Smart Chunking**: Divides documents into meaningful sections
    2. **Multi-level Processing**: Analyzes text at different granularities
    3. **Hierarchical Synthesis**: Combines information into a coherent whole
    
    The application is built with a modular design that allows for easy extension
    with new capabilities through the "passes" system.
    
    ## Technologies
    
    - **Backend**: Python with AsyncIO for efficient processing
    - **Frontend**: Streamlit for a simple, interactive user interface
    - **AI**: OpenAI's GPT models for text analysis and generation
    
    ## Getting Started
    
    1. Navigate to the Summarization page using the sidebar
    2. Upload a document or paste text
    3. Adjust processing options as needed
    4. Generate your enhanced notes
    """)

# Check for API key
if not api_key:
    st.warning(
        "⚠️ OpenAI API key not found! Please set the OPENAI_API_KEY environment variable "
        "before using the application."
    )

# Display version
st.sidebar.markdown("---")
st.sidebar.caption("Better Notes v0.1.0")