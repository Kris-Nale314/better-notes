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


# Check for OpenAI API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    import sys
    print("⚠️  WARNING: OPENAI_API_KEY environment variable not found.")
    print("Please set your API key in a .env file or environment variable.")

# Create necessary directories
for directory in ["data", "outputs", ".cache", "passes/configurations"]:
    Path(directory).mkdir(exist_ok=True, parents=True)

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
- **Identify Issues and Opportunities** to improve decision making
- **Extract Action Items** to ensure follow-through
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

    st.markdown("""
    ### 💡 Opportunity Identification
    
    Discover potential improvements, innovations and
    possibilities mentioned or implied in your documents.
    
    [Open Opportunity Identification](/Opportunity_Identification)
    """)

with col2:
    st.markdown("""
    ### 🔍 Issue Identification
    
    Extract problems, challenges, and areas of concern 
    from your documents to focus improvement efforts.
    
    [Open Issue Identification](/Issue_Identification)
    """)
    
    st.markdown("""
    ### ✅ Action Item Extraction
    
    Identify tasks, assignments, commitments, and follow-up
    items to ensure nothing falls through the cracks.
    
    [Open Action Items](/Action_Items)
    """)

# App information in expander
with st.expander("About Better Notes"):
    st.markdown("""
    ## How It Works
    
    Better Notes uses a lean architecture with these key components:
    
    1. **Smart Chunking**: Divides documents into meaningful sections
    2. **Multi-level Processing**: Analyzes text at different granularities
    3. **Specialized Passes**: Applies targeted analysis for different information types
    4. **Hierarchical Synthesis**: Combines information into a coherent whole
    
    The application is built with a modular design that makes it easy to extend
    with new capabilities through the "passes" system.
    
    ## Technologies
    
    - **Backend**: Python with AsyncIO for efficient processing
    - **Frontend**: Streamlit for a simple, interactive user interface
    - **AI**: OpenAI's GPT models for text analysis and generation
    - **Architecture**: Lean, modular design with specialized processors
    
    ## Getting Started
    
    1. Select a processing type from the sidebar navigation
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
st.sidebar.caption("Better Notes v0.1.1")