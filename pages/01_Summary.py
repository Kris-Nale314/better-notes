"""
Document Summarization Page for Better Notes.
Focuses on core summarization functionality with enhanced UI.
"""

import os
import time
import streamlit as st
from pathlib import Path
import tempfile
import datetime

# Import lean components
from lean.options import ProcessingOptions
from lean.orchestrator import SummarizerFactory
from lean.async_openai_adapter import AsyncOpenAIAdapter

# Import UI enhancements
from ui_utils.ui_enhance import (
    apply_custom_css, 
    enhance_result_display,
    display_chat_interface
)

# Configure page
st.set_page_config(
    page_title="Document Summarization - Better Notes",
    page_icon="📝",
    layout="wide"
)

# Apply custom CSS
apply_custom_css()

# Set up output directory structure
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "summaries").mkdir(exist_ok=True)

# --- Sidebar Configuration ---
st.sidebar.header("Summary Settings")

# Model selection
model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
selected_model = st.sidebar.selectbox("Language Model", model_options, index=0)

# Detail level
detail_options = ["essential", "detailed", "detailed-complex"]
detail_level = st.sidebar.selectbox(
    "Detail Level", 
    detail_options,
    format_func=lambda x: x.replace("-", " ").title(),
    help="Controls summary depth and structure"
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
    min_chunks = st.slider(
        "Minimum Chunks",
        min_value=1,
        max_value=10,
        value=3,
        help="Minimum number of sections to divide document into"
    )
    
    # Concurrency
    enable_parallel = st.checkbox("Parallel Processing", value=True)
    max_concurrent = st.slider(
        "Max Concurrent Tasks", 
        min_value=1, 
        max_value=10, 
        value=5,
        disabled=not enable_parallel
    )
    
    max_chunk_size = st.slider(
        "Max Chunk Size",
        min_value=500,
        max_value=3000,
        value=1500,
        step=100,
        help="Maximum size of each document chunk"
    )

# --- Main Content ---
st.title("📝 Document Summarization")
st.markdown("""
This tool creates rich, structured summaries of documents with customizable detail levels.
Upload a document and adjust settings to get the perfect summary for your needs.
""")

# Document upload
st.header("Upload Document")
upload_tab, paste_tab = st.tabs(["Upload File", "Paste Text"])

document_text = ""

with upload_tab:
    uploaded_file = st.file_uploader("Upload a text document", type=["txt", "md"])
        
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

with paste_tab:
    pasted_text = st.text_area(
        "Paste document text here",
        height=200,
        placeholder="Paste your document text here..."
    )
    if pasted_text:
        document_text = pasted_text
        st.success(f"Text loaded: {len(document_text)} characters")

# Warning for very large documents
if document_text and len(document_text) > 100000:
    st.warning("⚠️ Very large document detected. Consider splitting it into smaller sections for better results.")

# Custom instructions
with st.expander("Custom Instructions (Optional)", expanded=False):
    user_instructions = st.text_area(
        "Add specific instructions for the summary:",
        placeholder="E.g., 'Focus on technical details', 'Highlight financial implications', 'Emphasize methodology'",
        help="Your instructions will guide how the document is analyzed and summarized."
    )
    
    # Focus areas
    focus_areas = st.multiselect(
        "Focus Areas",
        options=["Key Findings", "Technical Details", "Financial Information", 
                 "Methodology", "Conclusions", "Recommendations", 
                 "Timeline", "Stakeholder Impact"],
        default=[],
        help="Select specific areas to emphasize in the summary"
    )
    
    # Add focus areas to instructions if selected
    if focus_areas and not "reanalysis_triggered" in st.session_state:
        if user_instructions:
            full_instructions = user_instructions + "\n\nFocus areas: " + ", ".join(focus_areas)
        else:
            full_instructions = "Focus areas: " + ", ".join(focus_areas)
        user_instructions = full_instructions

# Process button
process_button = st.button(
    "Generate Summary", 
    disabled=not document_text,
    type="primary",
    use_container_width=True
)

# Function to save output to file
def save_output_to_file(content: str) -> str:
    """Save summary output to the outputs folder with a timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"summary_{timestamp}.md"
    filepath = OUTPUT_DIR / "summaries" / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    return str(filepath)

# Initialize session state
if "summary_result" not in st.session_state:
    st.session_state.summary_result = None
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "processing_time" not in st.session_state:
    st.session_state.processing_time = 0
if "document_text" not in st.session_state:
    st.session_state.document_text = None
if "document_info" not in st.session_state:
    st.session_state.document_info = None
if "llm_client" not in st.session_state:
    st.session_state.llm_client = None

# Check if we're doing a reanalysis
if "reanalysis_settings" in st.session_state:
    reanalysis = st.session_state.reanalysis_settings
    detail_level = reanalysis["detail_level"]
    temperature = reanalysis["temperature"]
    user_instructions = reanalysis["user_instructions"]
    
    # Clear reanalysis settings for next run
    del st.session_state.reanalysis_settings
    st.session_state.reanalysis_triggered = True
    
    # Auto-process with new settings
    process_button = True

# Function to process document
def process_document():
    """Process the document and handle UI updates."""
    # Store document for later chat usage
    st.session_state.document_text = document_text
    
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
        # Create LLM client and store in session
        llm_client = AsyncOpenAIAdapter(
            model=selected_model,
            temperature=temperature
        )
        st.session_state.llm_client = llm_client
        
        # Create processing options
        options = ProcessingOptions(
            model_name=selected_model,
            temperature=temperature,
            min_chunks=min_chunks,
            max_chunk_size=max_chunk_size,
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
        
        # Store document info for chat
        if "document_info" in result:
            st.session_state.document_info = result["document_info"]
        
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
    
    # Handle error case
    if "error" in result:
        st.error(f"Error during processing: {result['error']}")
        if "metadata" in result:
            st.json(result["metadata"])
        st.stop()
    
    # Get the summary content
    if "summary" in result:
        summary_text = result["summary"]
        
        # Save to file
        saved_filepath = save_output_to_file(summary_text)
        
        # Show success message with filepath
        st.success(f"Summary generated in {st.session_state.processing_time:.2f} seconds and saved to {saved_filepath}")
        
        # Create tabs for different views of the results
        result_tabs = st.tabs(["Summary", "Chat with Document", "Adjust Summary", "Technical Info"])
        
        with result_tabs[0]:
            st.subheader("Document Summary")
            
            # Display executive summary if available in a highlighted box
            if "executive_summary" in result and result["executive_summary"]:
                st.markdown('<div class="executive-summary">', unsafe_allow_html=True)
                st.subheader("📋 Executive Summary")
                st.markdown(result["executive_summary"])
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")
            
            # Display main summary
            if summary_text:
                st.markdown(summary_text)
            else:
                st.warning("No summary was generated.")
            
            # Download option
            st.download_button(
                "Download Summary",
                data=summary_text,
                file_name=os.path.basename(saved_filepath),
                mime="text/markdown"
            )
        
        with result_tabs[1]:
            st.subheader("Chat about this Document")
            
            # Use the chat interface from ui_utils
            display_chat_interface(
                llm_client=st.session_state.llm_client,
                document_text=st.session_state.document_text,
                summary_text=summary_text,
                document_info=st.session_state.document_info
            )
        
        with result_tabs[2]:
            st.subheader("Adjust Summary Settings")
            
            # Create a form for reanalysis settings
            with st.form("reanalysis_form"):
                st.write("Adjust settings and provide new instructions to regenerate the summary")
                
                # Allow adjusting key parameters
                new_detail_level = st.select_slider(
                    "Detail Level",
                    options=["essential", "detailed", "detailed-complex"],
                    value=detail_level,
                    format_func=lambda x: x.replace("-", " ").title(),
                    help="Controls the depth of the summary"
                )
                
                new_temperature = st.slider(
                    "Temperature", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=temperature,
                    step=0.1,
                    help="Higher values = more creative, lower = more consistent"
                )
                
                # Expanded instructions area
                new_instructions = st.text_area(
                    "Summary Instructions",
                    value=user_instructions,
                    height=150,
                    help="Based on the initial summary, what would you like to change or emphasize?"
                )
                
                # Add specific focus options for summaries
                focus_options = st.multiselect(
                    "Focus Areas",
                    options=["Key Findings", "Technical Details", "Financial Information", 
                             "Methodology", "Conclusions", "Recommendations", 
                             "Timeline", "Stakeholder Impact"],
                    default=[],
                    help="Select specific areas to emphasize in the summary"
                )
                    
                # Add focus areas to instructions if selected
                if focus_options:
                    if new_instructions:
                        new_instructions += "\n\nFocus areas: " + ", ".join(focus_options)
                    else:
                        new_instructions = "Focus areas: " + ", ".join(focus_options)
                
                # Reanalysis button
                reanalyze_submitted = st.form_submit_button("Regenerate Summary", type="primary")
            
            if reanalyze_submitted:
                # Store new settings in session state
                st.session_state.reanalysis_settings = {
                    "detail_level": new_detail_level,
                    "temperature": new_temperature,
                    "user_instructions": new_instructions
                }
                
                # Reset processing flags
                st.session_state.processing_complete = False
                st.session_state.summary_result = None
                
                # Show processing message
                st.info("Starting summary generation with updated settings...")
                st.rerun()
        
        with result_tabs[3]:
            st.subheader("Technical Information")
            
            # Display metadata
            metadata = result.get("metadata", {})
            if metadata:
                with st.expander("Processing Details", expanded=True):
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Model", metadata.get("model", "Unknown"))
                    with cols[1]:
                        st.metric("Chunks", metadata.get("chunks_processed", 0))
                    with cols[2]:
                        st.metric("Detail Level", detail_level.replace("-", " ").title())
                    with cols[3]:
                        st.metric("Processing Time", f"{metadata.get('processing_time_seconds', 0):.2f}s")
            
            # Document stats if available
            doc_info = result.get("document_info", {})
            if doc_info:
                st.subheader("Document Statistics")
                
                # Check if basic_stats is available
                if "basic_stats" in doc_info:
                    stats = doc_info["basic_stats"]
                    # Create 3 columns for stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Word Count", stats.get("word_count", 0))
                        st.metric("Paragraphs", stats.get("paragraph_count", 0))
                    with col2:
                        st.metric("Sentences", stats.get("sentence_count", 0))
                        st.metric("Characters", stats.get("char_count", 0))
                    with col3:
                        st.metric("Est. Tokens", stats.get("estimated_tokens", 0))
                
                # Document classification
                is_transcript = doc_info.get("is_meeting_transcript", False)
                st.info(f"Document Type: {'Meeting Transcript' if is_transcript else 'Document'}")
                
                # Preview analysis if available
                if "preview_analysis" in doc_info:
                    preview = doc_info["preview_analysis"]
                    
                    if "key_topics" in preview and preview["key_topics"]:
                        st.write("Key Topics:")
                        for topic in preview["key_topics"]:
                            st.markdown(f"- {topic}")
                    
                    if "domain_categories" in preview and preview["domain_categories"]:
                        st.write("Domain Categories:")
                        for domain in preview["domain_categories"]:
                            st.markdown(f"- {domain}")

# When no document is loaded, show placeholder
if not document_text and not st.session_state.processing_complete:
    st.info("Upload a document to begin summarization.")