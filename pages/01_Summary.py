"""
Document Summarization Page for Better Notes.
Focuses on core summarization functionality with clean UI.
"""

import os
import time
import streamlit as st
from pathlib import Path
import tempfile

# Import lean components
from lean.options import ProcessingOptions
from lean.orchestrator import SummarizerFactory

# Configure page
st.set_page_config(
    page_title="Document Summarization - Better Notes",
    page_icon="📝",
    layout="wide"
)

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")

# Model selection
model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
selected_model = st.sidebar.selectbox("Model", model_options, index=0)

# Detail level
detail_options = ["essential", "detailed", "detailed-complex"]
detail_level = st.sidebar.selectbox(
    "Detail Level", 
    detail_options,
    format_func=lambda x: x.replace("-", " ").title(),
    help="Controls how detailed the summary will be"
)

# Number of chunks
min_chunks = st.sidebar.slider(
    "Minimum Chunks",
    min_value=1,
    max_value=10,
    value=3,
    help="Minimum number of sections to divide the document into"
)

# Temperature
temperature = st.sidebar.slider(
    "Temperature", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.2,
    step=0.1,
    help="Lower values give more deterministic outputs, higher values more creative"
)

# Performance options
st.sidebar.header("Performance")
enable_parallel = st.sidebar.checkbox("Parallel Processing", value=True)
max_concurrent = st.sidebar.slider(
    "Max Concurrent Tasks", 
    min_value=1, 
    max_value=10, 
    value=5,
    disabled=not enable_parallel
)

# --- Main Content ---
st.title("Document Summarization")
st.markdown("""
Upload a text document to generate an organized, detailed summary.
Adjust settings in the sidebar to control processing.
""")

# Document upload
st.subheader("Upload Document")
uploaded_file = st.file_uploader("Upload a text document", type=["txt", "md"])
    
document_text = ""
if uploaded_file:
    # Create a temp file to store the upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    # Read the file
    try:
        with open(tmp_path, "r", encoding="utf-8", errors="replace") as f:
            document_text = f.read()
        
        st.success(f"File loaded: {uploaded_file.name}")
        
        # Display a preview
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
        # Clean up the temp file
        try:
            os.unlink(tmp_path)
        except:
            pass

# Custom instructions
st.subheader("Custom Instructions (Optional)")
user_instructions = st.text_area(
    "Add specific instructions for how you want the document processed:",
    placeholder="E.g., 'Focus on technical details', 'Highlight financial implications'",
    help="Your instructions will guide how the document is analyzed and summarized."
)

# Process button
process_button = st.button(
    "Generate Summary", 
    disabled=not document_text,
    type="primary",
    use_container_width=True
)

# Initialize session state
if "summary_result" not in st.session_state:
    st.session_state.summary_result = None
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "processing_time" not in st.session_state:
    st.session_state.processing_time = 0

# Function to process document
def process_document():
    """Process the document and handle UI updates."""
    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error("OpenAI API key not found! Please set the OPENAI_API_KEY environment variable.")
        return
    
    # Set up progress tracking
    progress_container = st.container()
    progress_bar = progress_container.progress(0)
    status_text = progress_container.empty()
    
    start_time = time.time()
    
    try:
        # Create processing options
        options = ProcessingOptions(
            model_name=selected_model,
            temperature=temperature,
            min_chunks=min_chunks,
            detail_level=detail_level,
            max_concurrent_chunks=max_concurrent if enable_parallel else 1,
            user_instructions=user_instructions if user_instructions else None,
            include_metadata=True
        )
        
        # Create pipeline
        pipeline = SummarizerFactory.create_pipeline(api_key=api_key, options=options)
        orchestrator = pipeline['orchestrator']
        
        # Progress callback
        def update_progress(progress, message):
            progress_bar.progress(progress)
            status_text.text(message)
        
        # Process document
        result = orchestrator.process_document_sync(
            document_text,
            progress_callback=update_progress
        )
        
        # Store results
        st.session_state.summary_result = result
        st.session_state.processing_complete = True
        st.session_state.processing_time = time.time() - start_time
        
        # Clear progress indicators
        progress_container.empty()
        
        # Rerun to display results
        st.rerun()
        
    except Exception as e:
        progress_container.empty()
        st.error(f"Error processing document: {str(e)}")
        st.exception(e)

# Process the document when button is clicked
if process_button:
    st.session_state.processing_complete = False
    st.session_state.summary_result = None
    process_document()

# Display results
if st.session_state.processing_complete and st.session_state.summary_result:
    result = st.session_state.summary_result
    
    # Show success message
    st.success(f"Summary generated in {st.session_state.processing_time:.2f} seconds")
    
    # Handle error case
    if "error" in result:
        st.error(f"Error during processing: {result['error']}")
        if "metadata" in result:
            st.json(result["metadata"])
        st.stop()
    
    # Create tabs for different parts of the results
    summary_tab, metadata_tab = st.tabs(["Summary", "Metadata"])
    
    with summary_tab:
        # Display executive summary if available
        if "executive_summary" in result and result["executive_summary"]:
            st.subheader("Executive Summary")
            st.markdown(result["executive_summary"])
            st.markdown("---")
        
        # Display main summary
        st.subheader("Detailed Summary")
        if "summary" in result and result["summary"]:
            st.markdown(result["summary"])
        else:
            st.warning("No summary was generated.")
        
        # Refinement options
        if "summary" in result:
            st.subheader("Refine Summary")
            refine_cols = st.columns(2)
            
            with refine_cols[0]:
                if st.button("Add More Detail", use_container_width=True):
                    with st.spinner("Adding more detail..."):
                        # Get the same orchestrator
                        options = ProcessingOptions(model_name=selected_model)
                        pipeline = SummarizerFactory.create_pipeline(
                            api_key=os.environ.get("OPENAI_API_KEY", ""),
                            options=options
                        )
                        orchestrator = pipeline['orchestrator']
                        
                        # Refine the summary
                        refined_result = orchestrator.refine_summary_sync(
                            result, "more_detail"
                        )
                        
                        # Update session state
                        st.session_state.summary_result = refined_result
                        st.rerun()
            
            with refine_cols[1]:
                if st.button("Make More Concise", use_container_width=True):
                    with st.spinner("Making more concise..."):
                        # Get the same orchestrator
                        options = ProcessingOptions(model_name=selected_model)
                        pipeline = SummarizerFactory.create_pipeline(
                            api_key=os.environ.get("OPENAI_API_KEY", ""),
                            options=options
                        )
                        orchestrator = pipeline['orchestrator']
                        
                        # Refine the summary
                        refined_result = orchestrator.refine_summary_sync(
                            result, "more_concise"
                        )
                        
                        # Update session state
                        st.session_state.summary_result = refined_result
                        st.rerun()
        
        # Download option
        if "summary" in result:
            st.download_button(
                "Download Summary",
                data=result["summary"],
                file_name="summary.md",
                mime="text/markdown"
            )
    
    with metadata_tab:
        # Display metadata
        st.subheader("Processing Details")
        metadata = result.get("metadata", {})
        if metadata:
            cols = st.columns(3)
            with cols[0]:
                st.metric("Model", metadata.get("model", "Unknown"))
            with cols[1]:
                st.metric("Chunks", metadata.get("chunks_processed", 0))
            with cols[2]:
                st.metric("Processing Time", f"{metadata.get('processing_time_seconds', 0):.2f}s")
            
            # Display other metadata
            st.json({k: v for k, v in metadata.items() 
                    if k not in ["model", "chunks_processed", "processing_time_seconds"]})
        
        # Display document info
        st.subheader("Document Information")
        doc_info = result.get("document_info", {})
        if doc_info:
            # Display transcript status
            is_transcript = doc_info.get("is_meeting_transcript", False)
            st.info(f"Document Type: {'Meeting Transcript' if is_transcript else 'Document'}")
            
            # Display main document info, filtering out lengthy fields
            display_info = {k: v for k, v in doc_info.items() 
                           if k != "original_text_length" and not isinstance(v, (list, dict)) 
                           or (isinstance(v, (list, dict)) and len(str(v)) < 1000)}
            
            # Display as JSON
            st.json(display_info)

# When no document is loaded, show placeholder
if not document_text and not st.session_state.processing_complete:
    st.info("Upload a document to begin summarization.")