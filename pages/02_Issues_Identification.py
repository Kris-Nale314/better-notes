"""
Issues Identification Page - Better Notes
Identifies problems, challenges, and risks in documents using the enhanced architecture.
"""

import os
import sys
import time
import logging
import json
import traceback
from pathlib import Path
import tempfile
from typing import Dict, Any, Optional, List, Union

import streamlit as st
import asyncio

# Import core components
from orchestrator_factory import OrchestratorFactory
from config_manager import ConfigManager
from universal_llm_adapter import LLMAdapter

# Import UI utilities
from ui_utils.core_styling import apply_component_styles, apply_analysis_styles
from ui_utils.result_formatting import enhance_result_display, create_download_button
from ui_utils.chat_interface import display_chat_interface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("issues_analysis.log", mode="a")
    ]
)
logger = logging.getLogger("issues_identification")

# Configure page
st.set_page_config(
    page_title="Issues Identification - Better Notes",
    page_icon="⚠️",
    layout="wide"
)

# Apply styling
apply_component_styles()
apply_analysis_styles("issues")

# Setup output directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "issues").mkdir(exist_ok=True)

# Initialize session state
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "agent_result" not in st.session_state:
    st.session_state.agent_result = None
if "processing_time" not in st.session_state:
    st.session_state.processing_time = 0
if "document_info" not in st.session_state:
    st.session_state.document_info = None
if "document_text" not in st.session_state:
    st.session_state.document_text = ""
if "llm_client" not in st.session_state:
    st.session_state.llm_client = None
if "pipeline_status" not in st.session_state:
    st.session_state.pipeline_status = None
if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = []
if "detail_level" not in st.session_state:
    st.session_state.detail_level = "Standard"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.2
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 8
if "user_instructions" not in st.session_state:
    st.session_state.user_instructions = ""
if "focus_areas" not in st.session_state:
    st.session_state.focus_areas = []
if "enable_reviewer" not in st.session_state:
    st.session_state.enable_reviewer = True

# Main title and description
st.title("⚠️ Issues Identification")
st.markdown("""
This tool analyzes documents to identify issues, problems, risks, and challenges.
It uses a team of specialized AI agents coordinated by a Planner agent to extract, evaluate, and organize issues by severity.
""")

# Sidebar configuration
st.sidebar.header("Analysis Settings")

# Model selection
model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
selected_model = st.sidebar.selectbox("Language Model", model_options, index=0)

# Detail level
detail_level = st.sidebar.select_slider(
    "Detail Level",
    options=["Essential", "Standard", "Comprehensive"],
    value=st.session_state.detail_level,
    help="Controls the depth of analysis"
)
st.session_state.detail_level = detail_level

# Temperature
temperature = st.sidebar.slider(
    "Temperature", 
    min_value=0.0, 
    max_value=1.0, 
    value=st.session_state.temperature,
    step=0.1,
    help="Lower = more consistent, higher = more creative"
)
st.session_state.temperature = temperature

# Advanced settings
with st.sidebar.expander("Advanced Settings"):
    # Number of chunks
    num_chunks = st.slider(
        "Number of Document Chunks",
        min_value=3,
        max_value=20,
        value=st.session_state.num_chunks,
        help="Number of sections to divide document into"
    )
    st.session_state.num_chunks = num_chunks
    
    max_rpm = st.slider(
        "Max Requests Per Minute",
        min_value=5,
        max_value=30,
        value=10,
        step=1,
        help="Controls API request rate"
    )
    
    show_agent_details = st.checkbox(
        "Show Agent Interactions", 
        value=False,
        help="Display detailed agent activity logs"
    )
    
    enable_reviewer = st.checkbox(
        "Enable Review Step", 
        value=st.session_state.enable_reviewer,
        help="Final quality check of the analysis before delivery"
    )
    st.session_state.enable_reviewer = enable_reviewer
    
    debug_mode = st.checkbox(
        "Debug Mode",
        value=False,
        help="Show raw outputs and intermediate results for troubleshooting"
    )

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
            logger.error(f"File upload error: {str(e)}")
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass

with paste_tab:
    pasted_text = st.text_area(
        "Paste document text here",
        height=200,
        placeholder="Paste your document text here..."
    )
    if pasted_text:
        document_text = pasted_text
        st.session_state.document_text = document_text
        st.success(f"Text loaded: {len(document_text)} characters")

# Show document length and chunk size
if document_text:
    estimated_chunk_size = len(document_text) // num_chunks
    st.info(f"Document length: {len(document_text)} characters. With {num_chunks} chunks, each chunk will be approximately {estimated_chunk_size} characters.")

# Custom instructions
with st.expander("Custom Instructions (Optional)", expanded=False):
    user_instructions = st.text_area(
        "Add specific instructions for the analysis:",
        value=st.session_state.user_instructions,
        placeholder="E.g., 'Focus on technical issues', 'Prioritize security risks', 'Look for budget concerns'",
        help="Your instructions will guide how the agents analyze the document."
    )
    st.session_state.user_instructions = user_instructions
    
    # Focus areas
    focus_areas = st.multiselect(
        "Focus Areas",
        options=["Technical", "Process", "Resource", "Quality", "Risk"],
        default=st.session_state.focus_areas,
        help="Select specific types of issues to emphasize in the analysis"
    )
    st.session_state.focus_areas = focus_areas

# Process button
process_button = st.button(
    "Identify Issues", 
    disabled=not document_text,
    type="primary",
    use_container_width=True
)

# Display pipeline status function
def display_pipeline_status(pipeline_status):
    """Display the current pipeline status with an improved visualization."""
    if not pipeline_status:
        return
    
    stages = pipeline_status.get("stages", {})
    current = pipeline_status.get("current_stage")
    progress = pipeline_status.get("progress", 0)
    processing_time = pipeline_status.get("processing_time", 0)
    
    # Create timeline visualization
    st.markdown("""
    <style>
    .pipeline-container {
        display: flex;
        flex-direction: column;
        gap: 5px;
        padding: 10px;
        background: rgba(0,0,0,0.05);
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .pipeline-timeline {
        display: flex;
        width: 100%;
        height: 10px;
        background: rgba(0,0,0,0.1);
        border-radius: 5px;
        overflow: hidden;
        margin-bottom: 10px;
    }
    .timeline-progress {
        height: 100%;
        background: linear-gradient(90deg, #4e8cff, #20c997);
        border-radius: 5px;
    }
    .stage-row {
        display: flex;
        align-items: center;
        margin: 3px 0;
    }
    .stage-name {
        width: 150px;
        font-weight: 500;
    }
    .stage-bar {
        flex-grow: 1;
        height: 25px;
        background: rgba(0,0,0,0.1);
        border-radius: 5px;
        position: relative;
        overflow: hidden;
    }
    .stage-progress {
        height: 100%;
        background: #4e8cff;
        border-radius: 5px;
    }
    .stage-time {
        width: 80px;
        text-align: right;
        font-size: 0.8rem;
        color: rgba(0,0,0,0.6);
        padding-left: 10px;
    }
    .stage-status {
        width: 30px;
        text-align: center;
        font-size: 1.2rem;
        padding-right: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create pipeline container
    pipeline_html = [
        '<div class="pipeline-container">',
        '<div class="pipeline-timeline">',
        f'<div class="timeline-progress" style="width: {progress * 100}%;"></div>',
        '</div>'
    ]
    
    # Add each stage
    stage_names = [
        "document_analysis", "chunking", "planning", 
        "extraction", "aggregation", "evaluation", 
        "formatting", "review"
    ]
    
    for stage_name in stage_names:
        stage = stages.get(stage_name, {})
        status = stage.get("status", "waiting")
        
        # Calculate duration
        duration = stage.get("duration", 0)
        duration_str = f"{duration:.1f}s" if duration else ""
        
        # Determine icon based on status
        icon = "⏳"
        if status == "completed":
            icon = "✅"
        elif status == "running":
            icon = "🔄"
        elif status == "failed":
            icon = "❌"
        elif status == "skipped":
            icon = "⏭️"
        
        # Determine progress width
        progress_width = 100 if status == "completed" else stage.get("progress", 0) * 100
        
        # Add stage row
        pipeline_html.append(f"""
        <div class="stage-row">
            <div class="stage-status">{icon}</div>
            <div class="stage-name">{stage_name.replace('_', ' ').title()}</div>
            <div class="stage-bar">
                <div class="stage-progress" style="width: {progress_width}%;"></div>
            </div>
            <div class="stage-time">{duration_str}</div>
        </div>
        """)
    
    # Close container
    pipeline_html.append('</div>')
    
    # Render pipeline
    st.markdown("\n".join(pipeline_html), unsafe_allow_html=True)
    
    # Add overall stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Processing Time", f"{processing_time:.2f}s")
    with col2:
        st.metric("Current Stage", current.replace('_', ' ').title() if current else "Complete")
    with col3:
        st.metric("Progress", f"{progress * 100:.0f}%")
    
    # Show errors if any
    errors = pipeline_status.get("errors", [])
    if errors:
        with st.expander(f"Errors ({len(errors)})", expanded=False):
            for error in errors:
                st.error(f"{error.get('stage', 'Unknown stage')}: {error.get('message', 'Unknown error')}")

# Helper function to extract HTML content
def extract_html_content(result: Any) -> str:
    """
    Extract HTML content from different result formats.
    
    Args:
        result: Result object (dict, string, or other)
        
    Returns:
        HTML string or formatted string representation
    """
    # If result is a string, check if it's HTML-like
    if isinstance(result, str):
        if ("<html" in result.lower() or "<div" in result.lower() or 
            "<h1" in result.lower() or "<p>" in result.lower()):
            return result
    
    # If result is a dict, try to find HTML content
    if isinstance(result, dict):
        # Check for formatted_report field
        if "formatted_report" in result and isinstance(result["formatted_report"], str):
            return result["formatted_report"]
        
        # Look for HTML in any string field
        for key, value in result.items():
            if isinstance(value, str) and ("<html" in value.lower() or "<div" in value.lower()):
                return value
    
    # If no HTML found, convert to string representation
    if isinstance(result, dict):
        try:
            return json.dumps(result, indent=2)
        except:
            return str(result)
    
    return str(result)

# Function to save output to file
def save_output_to_file(content: Any) -> str:
    """
    Save analysis output to the outputs folder with a timestamp.
    Handles different content types.
    
    Args:
        content: Content to save (string, dict, or other)
        
    Returns:
        Path to saved file
    """
    # Extract HTML content if needed
    if not isinstance(content, str):
        content_str = extract_html_content(content)
    else:
        content_str = content
    
    # Generate filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"issues_{timestamp}.html"
    filepath = OUTPUT_DIR / "issues" / filename
    
    # Save to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content_str)
    
    return str(filepath)

# Function to process document
def process_document():
    """Process the document with our enhanced architecture."""
    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error("OpenAI API key not found! Please set the OPENAI_API_KEY environment variable.")
        return
    
    logger.info(f"Starting issues analysis with model: {selected_model}")
    
    # Create progress placeholders
    progress_container = st.container()
    progress_bar = progress_container.progress(0.0, text="Initializing...")
    status_text = progress_container.empty()
    pipeline_display = progress_container.empty()
    
    # Log container (if enabled)
    log_display = None
    if show_agent_details:
        log_display = st.empty()
        st.session_state.agent_logs = []
    
    # Start time tracking
    start_time = time.time()
    
    # Enhanced progress callback that updates all UI elements
    def update_progress(progress, message):
        """Update progress indicators in place."""
        # Update progress bar
        progress_bar.progress(progress, text=message)
        
        # Filter out chunk-level messages for cleaner UI
        is_chunk_message = any(term in message.lower() for term in 
                              ['chunk', 'processing chunk', 'extracting from chunk'])
        
        # Update status text for high-level messages
        if not is_chunk_message:
            status_text.text(message)
        
        # Add to logs if detailed logging is enabled
        if show_agent_details and log_display:
            if not is_chunk_message:
                # Add the log entry
                timestamp = time.strftime("%H:%M:%S")
                log_entry = f"[{timestamp}] {message}"
                st.session_state.agent_logs.append(log_entry)
                
                # Format logs
                log_html = []
                for entry in st.session_state.agent_logs[-20:]:  # Show last 20 logs
                    entry_class = "log-entry"
                    if "error" in entry.lower() or "failed" in entry.lower():
                        entry_class += " status-error"
                    elif "complete" in entry.lower() or "success" in entry.lower():
                        entry_class += " status-complete"
                        
                    log_html.append(f'<div class="{entry_class}">{entry}</div>')
                
                # Update the log display
                log_display.markdown("\n".join(log_html), unsafe_allow_html=True)
    
    try:
        # Map UI detail level to config values
        detail_map = {
            "Essential": "essential",
            "Standard": "standard",
            "Comprehensive": "comprehensive"
        }
        
        # Create configuration manager
        config_manager = ConfigManager()
        
        # Create the orchestrator using our factory
        orchestrator = OrchestratorFactory.create_orchestrator(
            implementation="langchain",  
            api_key=api_key,
            model=selected_model,
            temperature=temperature,
            max_chunk_size=len(document_text) // num_chunks if num_chunks > 0 else 10000,
            max_rpm=max_rpm,
            verbose=show_agent_details,
            config_manager=config_manager
        )
        
        # Enable debug mode if selected
        if debug_mode and hasattr(orchestrator, 'set_debug_mode'):
            orchestrator.set_debug_mode(True)
        
        # Store LLM client for chat interface
        if not st.session_state.llm_client:
            # Create a direct LLMAdapter for chat
            st.session_state.llm_client = LLMAdapter(
                api_key=api_key,
                model=selected_model,
                temperature=temperature
            )
        
        # Create processing options
        options = {
            "model_name": selected_model,
            "temperature": temperature,
            "crew_type": "issues",  # Explicitly set crew_type for config matching
            "min_chunks": num_chunks,
            "max_chunk_size": len(document_text) // num_chunks if num_chunks > 0 else 10000,
            "max_rpm": max_rpm,
            "enable_reviewer": enable_reviewer,
            "detail_level": detail_map.get(detail_level, "standard"),
            "focus_areas": [area.lower() for area in focus_areas],
            "user_instructions": user_instructions
        }
        
        logger.info("Starting document processing with enhanced architecture")
        
        # Custom context extractor for monitoring
        def extract_context_status(result):
            """Extract pipeline status from result metadata"""
            if isinstance(result, dict) and "_metadata" in result:
                metadata = result["_metadata"]
                return {
                    "stages": metadata.get("stages", {}),
                    "current_stage": metadata.get("current_stage", None),
                    "progress": metadata.get("progress", 1.0),
                    "progress_message": "Analysis complete",
                    "errors": metadata.get("errors", []),
                    "warnings": metadata.get("warnings", []),
                    "processing_time": metadata.get("processing_time", time.time() - start_time),
                    "run_id": metadata.get("run_id", "unknown")
                }
            return None
        
        # Process document with progress tracking
        result = orchestrator.process_document_sync(
            document_text,
            options=options,
            progress_callback=update_progress
        )
        
        # Extract pipeline status for UI display
        pipeline_status = extract_context_status(result)
        if pipeline_status:
            st.session_state.pipeline_status = pipeline_status
            pipeline_display.empty()  # Clear placeholder
            display_pipeline_status(pipeline_status)
        
        # Show raw result in debug mode
        if debug_mode:
            with st.expander("Debug: Raw Result Structure", expanded=False):
                st.json(result)
                
            # Also show pipeline explanation if the method exists
            if hasattr(orchestrator, 'explain_pipeline'):
                with st.expander("Debug: Pipeline Explanation", expanded=False):
                    explanation = orchestrator.explain_pipeline("issues")
                    st.json(explanation)
        
        # Store document info for chat
        if "_metadata" in result and "document_info" in result["_metadata"]:
            st.session_state.document_info = result["_metadata"]["document_info"]
        
        # Store results in session state
        st.session_state.agent_result = result
        st.session_state.processing_complete = True
        st.session_state.processing_time = time.time() - start_time
        
        # Clear progress display (pipeline display already handled)
        progress_bar.empty()
        status_text.empty()
        if log_display:
            log_display.empty()
        
        # Display results
        display_results(st.session_state.agent_result, st.session_state.processing_time)
        
    except Exception as e:
        # Clear progress display
        progress_bar.empty()
        status_text.empty()
        pipeline_display.empty()
        if log_display:
            log_display.empty()
        
        # Log the error
        logger.error(f"Error processing document: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Display error
        st.error(f"Error processing document: {str(e)}")
        
        with st.expander("Technical Error Details"):
            st.code(traceback.format_exc())
        
        # Update session state
        st.session_state.processing_complete = False
        st.session_state.agent_result = {"error": str(e)}

# Function to display results
def display_results(result, processing_time):
    """Display the processing results with improved result handling."""
    # Extract the HTML content or formatted result
    result_text = extract_html_content(result)
    
    # Handle error results
    if isinstance(result, dict) and "error" in result:
        st.error(f"Error during processing: {result['error']}")
        return
    
    # Save the result to a file
    saved_filepath = save_output_to_file(result_text)
    
    # Show success message
    st.success(f"Analysis completed in {processing_time:.2f} seconds")
    
    # Extract review information
    review_result = None
    if isinstance(result, dict) and "review_result" in result:
        review_result = result["review_result"]
    
    # Create tabs for different views
    result_tabs = st.tabs(["Report", "Chat with Document", "Adjust Analysis", "Technical Info"])
    
    with result_tabs[0]:
        st.subheader("Issues Identification Results")
        
        # Add review feedback if available
        if review_result and isinstance(review_result, dict):
            # Show assessment scores if available
            assessment = review_result.get("assessment", {})
            if assessment and isinstance(assessment, dict):
                st.info("Analysis Quality Assessment")
                
                # Count metrics to arrange in columns
                metrics = [(k.replace("_score", "").replace("_", " ").title(), v) 
                          for k, v in assessment.items() 
                          if isinstance(v, (int, float))]
                
                if metrics:
                    metric_cols = st.columns(len(metrics))
                    for i, (key, value) in enumerate(metrics):
                        metric_cols[i].metric(key, f"{value}/5")
                
                # Show summary assessment
                if "summary" in review_result:
                    with st.expander("Review Feedback", expanded=False):
                        st.markdown(f"**{review_result['summary']}**")
                        
                        # Show improvement suggestions if available
                        suggestions = review_result.get("improvement_suggestions", [])
                        if suggestions and isinstance(suggestions, list):
                            st.markdown("### Improvement Suggestions")
                            for suggestion in suggestions:
                                if isinstance(suggestion, dict):
                                    st.markdown(f"- **{suggestion.get('area', 'General')}**: {suggestion.get('suggestion', '')}")
        
        # Display enhanced result
        enhanced_html = enhance_result_display(result_text, "issues", st.session_state.detail_level.lower())
        st.markdown(enhanced_html, unsafe_allow_html=True)
        
        # Download option
        st.divider()
        create_download_button(result_text, os.path.basename(saved_filepath))
    
    with result_tabs[1]:
        st.subheader("Chat about this Issues Report")
        
        # Use the chat interface
        display_chat_interface(
            llm_client=st.session_state.llm_client,
            document_text=document_text,
            summary_text=result_text,
            document_info=st.session_state.document_info
        )
    
    with result_tabs[2]:
        st.subheader("Adjust Analysis Settings")
        
        # Create a form for reanalysis
        with st.form("reanalysis_form"):
            st.write("Adjust settings and provide new instructions to reanalyze the document")
            
            # Allow adjusting key parameters
            new_detail_level = st.select_slider(
                "Detail Level",
                options=["Essential", "Standard", "Comprehensive"],
                value=st.session_state.detail_level,
                help="Controls the depth of the analysis"
            )
            
            new_temperature = st.slider(
                "Temperature", 
                min_value=0.0, 
                max_value=1.0, 
                value=st.session_state.temperature,
                step=0.1,
                help="Higher values = more creative, lower = more consistent"
            )
            
            new_num_chunks = st.slider(
                "Number of Document Chunks",
                min_value=3,
                max_value=20,
                value=st.session_state.num_chunks,
                help="Number of sections to divide document into"
            )
            
            # Instructions area
            new_instructions = st.text_area(
                "Analysis Instructions",
                value=st.session_state.user_instructions,
                height=100,
                help="Based on the chat, what specific aspects would you like the analysis to focus on?"
            )
            
            # Focus options for issues
            new_focus_areas = st.multiselect(
                "Focus Areas",
                options=["Technical", "Process", "Resource", "Quality", "Risk"],
                default=st.session_state.focus_areas,
                help="Select specific types of issues to emphasize"
            )
            
            # Reviewer option
            new_enable_reviewer = st.checkbox(
                "Enable Review Step",
                value=st.session_state.enable_reviewer,
                help="Final quality check of the analysis"
            )
            
            # Reanalysis button
            reanalyze_submitted = st.form_submit_button("Reanalyze Document", type="primary")
        
        if reanalyze_submitted:
            # Update session state variables for the next run
            st.session_state.detail_level = new_detail_level
            st.session_state.temperature = new_temperature
            st.session_state.num_chunks = new_num_chunks
            st.session_state.user_instructions = new_instructions
            st.session_state.focus_areas = new_focus_areas
            st.session_state.enable_reviewer = new_enable_reviewer
            
            # Reset processing status
            st.session_state.processing_complete = False
            st.session_state.agent_result = None
            st.session_state.pipeline_status = None
            
            # Show processing message
            st.info("Starting reanalysis with updated settings...")
            
            # Process document
            process_document()
    
    with result_tabs[3]:
        st.subheader("Technical Information")
        
        # Show pipeline status if available
        if st.session_state.pipeline_status:
            st.markdown("### Pipeline Status")
            display_pipeline_status(st.session_state.pipeline_status)
        
        # Document stats if available
        document_info = st.session_state.document_info
        if document_info and isinstance(document_info, dict) and "basic_stats" in document_info:
            stats = document_info["basic_stats"]
            st.markdown("### Document Statistics")
            
            # Create columns for stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Word Count", stats.get("word_count", 0))
                st.metric("Paragraphs", stats.get("paragraph_count", 0))
            with col2:
                st.metric("Sentences", stats.get("sentence_count", 0))
                st.metric("Characters", stats.get("char_count", 0))
            with col3:
                st.metric("Est. Tokens", stats.get("estimated_tokens", 0))
                st.metric("Chunks Processed", st.session_state.num_chunks)
        
        # Technical configuration details
        st.markdown("### Analysis Configuration")
        tech_details = {
            "model_used": selected_model,
            "temperature": st.session_state.temperature,
            "detail_level": st.session_state.detail_level,
            "document_length": len(document_text),
            "number_of_chunks": st.session_state.num_chunks,
            "calculated_chunk_size": len(document_text) // st.session_state.num_chunks,
            "max_rpm": max_rpm,
            "processing_time_seconds": round(processing_time, 2),
            "output_file": saved_filepath,
            "reviewer_enabled": st.session_state.enable_reviewer,
            "user_preferences": {
                "user_instructions": st.session_state.user_instructions if st.session_state.user_instructions else "None provided",
                "focus_areas": st.session_state.focus_areas
            }
        }
        
        # Display in a collapsible section
        with st.expander("Configuration Details", expanded=False):
            st.json(tech_details)
        
        # Show plan if available
        if isinstance(result, dict) and "_metadata" in result and "plan" in result["_metadata"]:
            st.markdown("### Agent Plan")
            with st.expander("Planner-Generated Instructions", expanded=False):
                st.json(result["_metadata"]["plan"])
        
        # Additional metadata
        if isinstance(result, dict) and "_metadata" in result:
            metadata = result["_metadata"]
            # Remove plan and document_info to avoid duplication
            metadata_copy = metadata.copy()
            if "plan" in metadata_copy:
                del metadata_copy["plan"]
            if "document_info" in metadata_copy:
                del metadata_copy["document_info"]
                
            if metadata_copy:
                st.markdown("### Processing Metadata")
                with st.expander("Processing Stats", expanded=False):
                    st.json(metadata_copy)

# Run when process button is clicked
if process_button:
    # Reset processing flags
    st.session_state.processing_complete = False
    st.session_state.agent_result = None
    st.session_state.pipeline_status = None
    
    # Process the document
    process_document()

# If results are already available, display them
elif st.session_state.processing_complete and st.session_state.agent_result:
    display_results(st.session_state.agent_result, st.session_state.processing_time)
    
    # If pipeline status is available, display it
    if st.session_state.pipeline_status:
        with st.expander("Pipeline Execution Details", expanded=False):
            display_pipeline_status(st.session_state.pipeline_status)
            
# Add documentation at the bottom
with st.expander("About Issues Identification", expanded=False):
    st.markdown("""
    ### How It Works

    This tool uses a team of specialized AI agents to analyze your document:

    1. **Document Analysis & Chunking**: The document is analyzed and divided into manageable chunks
    2. **Planning**: The Planner agent creates tailored instructions for all other agents
    3. **Extraction**: Extractor agents identify potential issues from each document chunk
    4. **Aggregation**: Similar issues from different chunks are combined and deduplicated
    5. **Evaluation**: Each issue is assessed for severity, impact, and importance
    6. **Formatting**: Issues are organized into a clear, structured report
    7. **Review**: The completed analysis is evaluated for quality and alignment with your needs

    ### Tips for Better Results

    - Choose a more powerful model (GPT-4) for complex documents
    - Adjust the detail level based on how comprehensive you need the analysis to be
    - Use focus areas to emphasize specific types of issues
    - Provide custom instructions to guide the analysis towards your specific concerns
    """)