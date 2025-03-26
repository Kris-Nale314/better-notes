"""
Issues Analysis Page - Better Notes
Identifies problems, challenges, and risks in documents using the enhanced architecture.
Uses the new Orchestrator with ProcessingContext and IssuesCrew with integrated Planner.
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

# Import core components with new architecture
from orchestrator import Orchestrator, ProcessingContext
from orchestrator_factory import OrchestratorFactory
from config_manager import ConfigManager

# Import UI utilities
from ui_utils.core_styling import apply_component_styles, apply_analysis_styles
from ui_utils.result_formatting import enhance_result_display, create_download_button
from ui_utils.chat_interface import display_chat_interface
from ui_utils.progress_tracking import create_progress_callback

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
    value="Standard",
    help="Controls the depth of analysis"
)

# Temperature
temperature = st.sidebar.slider(
    "Temperature", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.2,
    step=0.1,
    help="Lower = more consistent, higher = more creative"
)

# Advanced settings
with st.sidebar.expander("Advanced Settings"):
    # Number of chunks
    num_chunks = st.slider(
        "Number of Document Chunks",
        min_value=3,
        max_value=20,
        value=8,
        help="Number of sections to divide document into"
    )
    
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
        value=True,
        help="Final quality check of the analysis before delivery"
    )
    
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
        placeholder="E.g., 'Focus on technical issues', 'Prioritize security risks', 'Look for budget concerns'",
        help="Your instructions will guide how the agents analyze the document."
    )
    
    # Focus areas
    focus_areas = st.multiselect(
        "Focus Areas",
        options=["Technical", "Process", "Resource", "Quality", "Risk"],
        default=[],
        help="Select specific types of issues to emphasize in the analysis"
    )

# Process button
process_button = st.button(
    "Identify Issues", 
    disabled=not document_text,
    type="primary",
    use_container_width=True
)

# Function to extract HTML content
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
    progress_placeholder = progress_container.empty()
    status_text_placeholder = progress_container.empty()
    
    # Create stage indicators using columns and placeholders
    stages = ["Planning", "Extraction", "Aggregation", "Evaluation", "Formatting"]
    if enable_reviewer:
        stages.append("Review")
    
    stage_cols = st.columns(len(stages))
    stage_indicators = []
    
    for i, stage in enumerate(stages):
        with stage_cols[i]:
            # Use placeholders for each stage indicator
            stage_indicators.append({
                "name": stage,
                "placeholder": st.empty(),
                "status": "waiting"
            })
            
            # Set initial waiting state
            stage_indicators[i]["placeholder"].markdown(f"""
                <div style="text-align: center;">
                    <div style="font-size: 1.5rem;">⏳</div>
                    <div style="font-weight: 500;">{stage}</div>
                    <div style="font-size: 0.8rem;">Waiting</div>
                </div>
            """, unsafe_allow_html=True)
    
    # Log container (if enabled)
    log_placeholder = None
    if show_agent_details:
        log_placeholder = st.empty()
        st.session_state.agent_logs = []
    
    # Track overall progress
    progress_value = 0.0
    start_time = time.time()
    
    # Progress callback that updates placeholders
    def update_progress(progress, message):
        """Update progress indicators in place."""
        nonlocal progress_value
        progress_value = progress
        
        # Update progress bar
        progress_placeholder.progress(progress)
        
        # Filter out chunk-level messages for cleaner UI
        is_chunk_message = any(term in message.lower() for term in 
                              ['chunk', 'processing chunk', 'extracting from chunk'])
        
        # Only update status text for high-level messages
        if not is_chunk_message:
            status_text_placeholder.text(message)
        
        # Determine current stage based on progress value
        current_stage = None
        if progress <= 0.2:
            current_stage = "Planning"
        elif progress <= 0.5:
            current_stage = "Extraction"
        elif progress <= 0.65:
            current_stage = "Aggregation"
        elif progress <= 0.75:
            current_stage = "Evaluation"
        elif progress <= 0.85:
            current_stage = "Formatting"
        elif progress <= 1.0 and enable_reviewer:
            current_stage = "Review"
        
        # Update stage indicators
        for i, indicator in enumerate(stage_indicators):
            stage = indicator["name"]
            status = indicator["status"]
            new_status = "waiting"
            
            # Determine new status
            if stage == current_stage:
                new_status = "working"
            elif stages.index(stage) < stages.index(current_stage) if current_stage else False:
                new_status = "complete"
                
            # Only update if status changed
            if new_status != status:
                stage_indicators[i]["status"] = new_status
                
                # Determine icon and color
                if new_status == "complete":
                    icon = "✅"
                    color = "#20c997"
                elif new_status == "working":
                    icon = "🔄"
                    color = "#ff9f43"
                else:
                    icon = "⏳"
                    color = "#6c757d"
                
                # Update the placeholder with new status
                indicator["placeholder"].markdown(f"""
                    <div style="text-align: center;">
                        <div style="font-size: 1.5rem;">{icon}</div>
                        <div style="font-weight: 500; color: {color};">{stage}</div>
                        <div style="font-size: 0.8rem;">{new_status.capitalize()}</div>
                    </div>
                """, unsafe_allow_html=True)
        
        # Add to logs if detailed logging is enabled
        if show_agent_details and log_placeholder:
            if not is_chunk_message or log_placeholder is None:
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
                
                # Update the log placeholder
                log_placeholder.markdown("\n".join(log_html), unsafe_allow_html=True)
    
    try:
        # Map UI detail level to config values
        detail_map = {
            "Essential": "essential",
            "Standard": "standard",
            "Comprehensive": "comprehensive"
        }
        
        # Create configuration manager
        config_manager = ConfigManager()
        
        # Create options dictionary
        options = {
            "crew_type": "issues",
            "detail_level": detail_map.get(detail_level, "standard"),
            "focus_areas": [area.lower() for area in focus_areas],
            "user_instructions": user_instructions,
            "min_chunks": num_chunks,
            "max_chunk_size": len(document_text) // num_chunks if num_chunks > 0 else 10000,
            "enable_reviewer": enable_reviewer
        }
        
        logger.info(f"Starting document processing with options: {json.dumps(options, default=str)}")
        
        # Create the orchestrator directly using the factory function
        orchestrator = OrchestratorFactory.create_orchestrator(
            api_key=api_key,
            model=selected_model,
            temperature=temperature,
            max_chunk_size=len(document_text) // num_chunks if num_chunks > 0 else 10000,
            max_rpm=max_rpm,
            verbose=show_agent_details,
            config_manager=config_manager
        )
        
        # Store LLM client for chat interface
        st.session_state.llm_client = orchestrator.llm_client
        
        # Process document with progress tracking using the synchronous method
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async method in the event loop
            result = loop.run_until_complete(
                orchestrator.process_document(
                    document_text,
                    options=options,
                    progress_callback=update_progress
                )
            )
        finally:
            # Always close the loop
            loop.close()
        
        # Show raw result in debug mode
        if debug_mode:
            with st.expander("Debug: Raw Result Structure", expanded=False):
                st.json(result)
        
        # Store document info for chat
        if "_metadata" in result and "document_info" in result["_metadata"]:
            st.session_state.document_info = result["_metadata"]["document_info"]
        
        # Store results in session state
        st.session_state.agent_result = result
        st.session_state.processing_complete = True
        st.session_state.processing_time = time.time() - start_time
        
        # Clear progress display
        progress_placeholder.empty()
        status_text_placeholder.empty()
        for indicator in stage_indicators:
            indicator["placeholder"].empty()
        if log_placeholder:
            log_placeholder.empty()
        
        # Display results
        display_results(st.session_state.agent_result, st.session_state.processing_time)
        
    except Exception as e:
        # Clear progress display
        progress_placeholder.empty()
        status_text_placeholder.empty()
        for indicator in stage_indicators:
            indicator["placeholder"].empty()
        if log_placeholder:
            log_placeholder.empty()
        
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
    """Display the processing results using existing formatting utilities."""
    # Handle error results
    if isinstance(result, dict) and "error" in result:
        st.error(f"Error during processing: {result['error']}")
        return
    
    # Extract the formatted report
    formatted_report = None
    if isinstance(result, dict) and "formatted_report" in result:
        formatted_report = result["formatted_report"]
    elif isinstance(result, str):
        formatted_report = result
    
    # Save the result to a file
    if formatted_report:
        saved_filepath = save_output_to_file(formatted_report)
    else:
        saved_filepath = save_output_to_file(result)
    
    # Show success message
    st.success(f"Analysis completed in {processing_time:.2f} seconds")
    
    # Extract review information
    review_result = None
    if isinstance(result, dict) and "review_result" in result:
        review_result = result["review_result"]
    
    # Create tabs for different views
    result_tabs = st.tabs(["Report", "Chat with Document", "Technical Info"])
    
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
        
        # Display formatted result using existing utilities
        content_to_display = formatted_report or result
        enhanced_content = enhance_result_display(content_to_display, "issues", detail_level.lower())
        st.markdown(enhanced_content, unsafe_allow_html=True)
        
        # Download option
        st.divider()
        create_download_button(result, os.path.basename(saved_filepath))
        
    
    with result_tabs[1]:
        st.subheader("Chat about this Issues Report")
        
        # Use the chat interface
        display_chat_interface(
            llm_client=st.session_state.llm_client,
            document_text=document_text,
            summary_text=formatted_report or json.dumps(result, indent=2),
            document_info=st.session_state.document_info
        )
    
    with result_tabs[2]:
        st.subheader("Technical Information")
        
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
                st.metric("Chunks Processed", num_chunks)
        
        # Technical configuration details
        st.markdown("### Analysis Configuration")
        tech_details = {
            "model_used": selected_model,
            "temperature": temperature,
            "detail_level": detail_level,
            "document_length": len(document_text),
            "number_of_chunks": num_chunks,
            "calculated_chunk_size": len(document_text) // num_chunks,
            "max_rpm": max_rpm,
            "processing_time_seconds": round(processing_time, 2),
            "output_file": saved_filepath,
            "reviewer_enabled": enable_reviewer,
            "user_preferences": {
                "user_instructions": user_instructions if user_instructions else "None provided",
                "focus_areas": focus_areas
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
    
    # Process the document
    process_document()

# If results are already available, display them
elif st.session_state.processing_complete and st.session_state.agent_result:
    display_results(st.session_state.agent_result, st.session_state.processing_time)