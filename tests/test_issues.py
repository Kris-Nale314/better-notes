"""
Test Issues Analysis - Simplified test script for the new architecture.
Demonstrates how to use the new components.
"""

import os
import time
import streamlit as st
from pathlib import Path
import tempfile
import json
import datetime
import traceback
import logging
from typing import Dict, Any, Optional

# Import our new components
from universal_llm_adapter import UniversalLLMAdapter
from config_manager import ConfigManager, ProcessingOptions
from orchestrator_factory import OrchestratorFactory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
   page_title="Issues Analysis Test",
   page_icon="🔍",
   layout="wide"
)

# Initialize session state
if "test_results" not in st.session_state:
   st.session_state.test_results = None
if "processing_time" not in st.session_state:
   st.session_state.processing_time = 0
if "document_text" not in st.session_state:
   st.session_state.document_text = ""
if "logs" not in st.session_state:
   st.session_state.logs = []

# Page header
st.title("🔍 Issues Analysis Test")
st.markdown("Test the new streamlined architecture for Better Notes")

# Sidebar configuration
st.sidebar.header("Test Configuration")

# Model selection
model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
selected_model = st.sidebar.selectbox("Language Model", model_options, index=0)

# Detail level
detail_level = st.sidebar.select_slider(
   "Detail Level",
   options=["Essential", "Standard", "Comprehensive"],
   value="Standard",
   help="Controls analysis depth"
)

# Temperature
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

# Number of chunks
num_chunks = st.sidebar.slider("Number of Chunks", 3, 15, 8)

# Focus areas
st.sidebar.subheader("Focus Areas")
focus_areas = []
if st.sidebar.checkbox("Technical"): focus_areas.append("Technical")
if st.sidebar.checkbox("Process"): focus_areas.append("Process")
if st.sidebar.checkbox("Resource"): focus_areas.append("Resource")
if st.sidebar.checkbox("Quality"): focus_areas.append("Quality")
if st.sidebar.checkbox("Risk"): focus_areas.append("Risk")

# Reviewer
enable_reviewer = st.sidebar.checkbox("Enable Review Step", value=True)

# Debug options
st.sidebar.subheader("Debug Options")
show_logs = st.sidebar.checkbox("Show Logs", value=True)
show_plan = st.sidebar.checkbox("Show Plan", value=True)
show_metadata = st.sidebar.checkbox("Show Metadata", value=True)

# Document upload
st.header("Upload Document")
upload_tab, paste_tab = st.tabs(["Upload File", "Paste Text"])

document_text = st.session_state.document_text

with upload_tab:
   uploaded_file = st.file_uploader("Upload a text document", type=["txt", "md"])
   if uploaded_file:
       with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
           tmp_file.write(uploaded_file.getvalue())
           tmp_path = tmp_file.name
       
       try:
           with open(tmp_path, "r", encoding="utf-8", errors="replace") as f:
               document_text = f.read()
           
           st.success(f"File loaded: {uploaded_file.name}")
           st.session_state.document_text = document_text
           
           with st.expander("Document Preview"):
               st.text_area(
                   "Content",
                   document_text[:2000] + ("..." if len(document_text) > 2000 else ""),
                   height=200,
                   disabled=True
               )
       except Exception as e:
           st.error(f"Error reading file: {str(e)}")
       finally:
           try:
               os.unlink(tmp_path)
           except:
               pass

with paste_tab:
   pasted_text = st.text_area("Paste document text here", height=200)
   if pasted_text:
       document_text = pasted_text
       st.session_state.document_text = document_text
       st.success(f"Text loaded: {len(document_text)} characters")

# Show document stats
if document_text:
   st.info(f"Document length: {len(document_text)} characters. With {num_chunks} chunks, each chunk will be approximately {len(document_text) // num_chunks} characters.")

# Custom instructions
with st.expander("Custom Instructions (Optional)"):
   user_instructions = st.text_area(
       "Add specific analysis instructions:",
       placeholder="E.g., 'Focus on technical issues', 'Prioritize security risks'",
       key="user_instructions"
   )

# Process button
process_button = st.button("Run Analysis", disabled=not document_text, type="primary", use_container_width=True)

# Progress indicators
progress_container = st.empty()
status_text = st.empty()

def run_analysis():
   """Run the issues analysis test using our new components."""
   st.session_state.logs = []
   
   # Create progress bar
   progress_bar = progress_container.progress(0)
   status_text.text("Starting analysis...")
   
   # Set up log capture
   if show_logs:
       log_capture = []
       class LogCaptureHandler(logging.Handler):
           def emit(self, record):
               log_capture.append(self.format(record))
               st.session_state.logs = log_capture[-100:]  # Keep last 100 logs
       
       log_handler = LogCaptureHandler()
       log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
       root_logger = logging.getLogger()
       root_logger.addHandler(log_handler)
   
   start_time = time.time()
   
   def update_progress(progress, message):
       """Update progress indicators."""
       progress_bar.progress(progress)
       status_text.text(message)
       st.session_state.logs.append(f"{time.strftime('%H:%M:%S')} - {message}")
   
   try:
       # Get API key
       api_key = os.environ.get("OPENAI_API_KEY")
       if not api_key:
           progress_container.empty()
           status_text.empty()
           st.error("OpenAI API key not found! Please set the OPENAI_API_KEY environment variable.")
           return
       
       # Create config manager
       update_progress(0.05, "Initializing components...")
       config_manager = ConfigManager()
       
       # Create processing options
       process_options = ProcessingOptions(
           model_name=selected_model,
           temperature=temperature,
           detail_level=detail_level.lower(),
           min_chunks=num_chunks,
           max_chunk_size=len(document_text) // num_chunks + 100,
           focus_areas=focus_areas,
           user_instructions=user_instructions,
           enable_reviewer=enable_reviewer,
           max_rpm=20
       )
       
       # Create orchestrator
       update_progress(0.1, "Creating orchestrator...")
       orchestrator = OrchestratorFactory.create_from_options(
           process_options,
           api_key=api_key,
           config_manager=config_manager
       )
       
       # Process document
       update_progress(0.15, "Starting document processing...")
       result = orchestrator.process_document(
           document_text,
           options=process_options,
           progress_callback=update_progress
       )
       
       # Save processing time
       end_time = time.time()
       st.session_state.processing_time = end_time - start_time
       
       # Store results
       st.session_state.test_results = result
       
       # Clear progress indicators
       progress_container.empty()
       status_text.empty()
       
       st.success(f"Analysis completed in {st.session_state.processing_time:.2f} seconds")
       
       # Remove log handler
       if show_logs:
           root_logger.removeHandler(log_handler)
       
   except Exception as e:
       # Display error
       progress_container.empty()
       status_text.empty()
       st.error(f"Error during analysis: {str(e)}")
       
       # Show traceback
       with st.expander("Error Details"):
           st.code(traceback.format_exc())
       
       # Remove log handler
       if show_logs:
           root_logger.removeHandler(log_handler)

def display_results():
   """Display the analysis results."""
   results = st.session_state.test_results
   processing_time = st.session_state.processing_time
   
   # Create tabs for different views
   result_tabs = st.tabs(["Report", "Plan & Metadata", "Logs", "Raw Data"])
   
   with result_tabs[0]:
       st.subheader("Issues Analysis Report")
       
       # Get issues result
       if isinstance(results, dict) and "issues" in results:
           issues_result = results["issues"]
           
           # Handle different result formats
           if isinstance(issues_result, dict) and "raw_output" in issues_result:
               st.markdown(issues_result["raw_output"], unsafe_allow_html=True)
           elif isinstance(issues_result, str):
               st.markdown(issues_result, unsafe_allow_html=True)
           else:
               st.json(issues_result)
           
           # Show review result
           if isinstance(issues_result, dict) and "review_result" in issues_result:
               st.subheader("Review Assessment")
               review = issues_result["review_result"]
               
               if isinstance(review, dict) and "assessment" in review:
                   cols = st.columns(len(review["assessment"]))
                   for i, (key, value) in enumerate(review["assessment"].items()):
                       cols[i].metric(key.replace("_score", "").title(), f"{value}/5")
               
               if isinstance(review, dict) and "summary" in review:
                   st.info(f"Review Summary: {review['summary']}")
       else:
           st.warning("No issues analysis results found")
   
   with result_tabs[1]:
       if show_plan:
           st.subheader("Agent Plan")
           
           # Try to find the plan in results
           plan = None
           if isinstance(results, dict) and "_metadata" in results and "plan" in results["_metadata"]:
               plan = results["_metadata"]["plan"]
           
           if plan:
               st.json(plan)
           else:
               st.info("No agent plan found in results")
       
       if show_metadata:
           st.subheader("Processing Metadata")
           
           # Try to find metadata in results
           metadata = None
           if isinstance(results, dict):
               if "_metadata" in results:
                   metadata = results["_metadata"]
           
           if metadata:
               st.json(metadata)
           else:
               st.info("No metadata found in results")
       
       # Processing stats
       st.subheader("Processing Statistics")
       col1, col2, col3 = st.columns(3)
       col1.metric("Processing Time", f"{processing_time:.2f}s")
       col2.metric("Document Length", f"{len(document_text):,} chars")
       col3.metric("Number of Chunks", f"{num_chunks}")
   
   with result_tabs[2]:
       st.subheader("Processing Logs")
       
       if show_logs and st.session_state.logs:
           for log in st.session_state.logs:
               st.text(log)
       else:
           st.info("No logs available. Enable 'Show Logs' to see detailed logs.")
   
   with result_tabs[3]:
       st.subheader("Raw Results Data")
       
       # Show the raw results
       st.json(results)
       
       # Download option
       if results:
           results_json = json.dumps(results, default=str, indent=2)
           st.download_button(
               "Download Results",
               data=results_json,
               file_name=f"issues_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
               mime="application/json"
           )

# Run when process button clicked
if process_button:
   run_analysis()

# Display results if available
if st.session_state.test_results:
   display_results()