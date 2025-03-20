"""
Issue Identification Page for Better Notes.
Identifies problems, challenges, and concerns in documents.
"""

import os
import time
import tempfile
import streamlit as st
from pathlib import Path

# Import lean components
from lean.options import ProcessingOptions
from lean.orchestrator import SummarizerFactory

# Import specialized issue handler (optional - can use core system instead)
from passes.issues import create_issue_identifier

# Configure page
st.set_page_config(
    page_title="Issue Identification - Better Notes",
    page_icon="🔍",
    layout="wide"
)

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")

# Model selection
model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
selected_model = st.sidebar.selectbox("Model", model_options, index=0)

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

# --- Issue-Specific Options ---
with st.sidebar.expander("Issue Analysis Options", expanded=True):
    severity_threshold_options = ["low", "medium", "high", "critical"]
    severity_threshold = st.selectbox(
        "Minimum Severity:",
        severity_threshold_options,
        index=1,  # Default to "medium"
        help="Only show issues at or above this severity level."
    )
    include_context = st.checkbox("Include Context", value=True, help="Show the surrounding text for each issue.")

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
enable_caching = st.sidebar.checkbox("Enable Caching", value=True)

# --- Main Content ---
st.title("Issue Identification")
st.markdown("""
Upload a document to identify problems, challenges, and areas of concern.
The system will analyze the content and extract explicit and implicit issues.
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

# Analysis focus
st.subheader("Analysis Focus (Optional)")
user_instructions = st.text_area(
    "Add specific focus areas for issue identification:",
    placeholder="E.g., 'Focus on technical risks', 'Identify financial concerns', 'Look for compliance issues'",
    help="Your instructions will guide what types of issues are identified."
)

# Process button
process_button = st.button(
    "Identify Issues",
    disabled=not document_text,
    type="primary",
    use_container_width=True
)

# Initialize session state
if "issue_result" not in st.session_state:
    st.session_state.issue_result = None
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "processing_time" not in st.session_state:
    st.session_state.processing_time = 0

def process_document():
    """Process document and update UI."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error("OpenAI API key not found! Please set the OPENAI_API_KEY environment variable.")
        return

    progress_container = st.container()
    progress_bar = progress_container.progress(0)
    status_text = progress_container.empty()

    start_time = time.time()

    try:
        # Create pass_options dictionary
        pass_options = {
            "issue_identification": {
                "severity_threshold": severity_threshold,
                "include_context": include_context
            }
        }

        # Create ProcessingOptions
        options = ProcessingOptions(
            model_name=selected_model,
            temperature=temperature,
            min_chunks=min_chunks,
            max_concurrent_chunks=max_concurrent if enable_parallel else 1,
            user_instructions=user_instructions if user_instructions else None,
            include_metadata=True,
            enable_caching=enable_caching,
            passes=["issue_identification"],
            pass_options=pass_options
        )

        # Progress callback
        def update_progress(progress, message):
            progress_bar.progress(progress)
            status_text.text(message)
        
        # APPROACH 1: Use standard orchestrator with passes
        pipeline = SummarizerFactory.create_pipeline(api_key=api_key, options=options)
        orchestrator = pipeline['orchestrator']
        
        # Process document
        result = orchestrator.process_document_sync(
            document_text,
            progress_callback=update_progress
        )
        
        # Reshape result for consistent handling
        if "passes" in result and "issue_identification" in result["passes"]:
            issue_results = result["passes"]["issue_identification"]
            if "result" in issue_results:
                # Keep the original result for reference
                result["original_result"] = result.copy()
                # Extract the issue data to top level for easier access
                result["issues"] = issue_results["result"].get("issues", [])
                result["issue_summary"] = issue_results["result"].get("summary", "")

        # APPROACH 2: Use specialized issue identifier (alternative)
        # Uncomment this section to use the specialized module instead
        """
        # Create LLM client
        llm_client = AsyncOpenAIAdapter(
            model=selected_model,
            api_key=api_key,
            temperature=temperature
        )
        
        # Create and use issue identifier
        issue_identifier = create_issue_identifier(llm_client, options)
        result = issue_identifier.process_document_sync(
            document_text,
            progress_callback=update_progress
        )
        """

        # Store results
        st.session_state.issue_result = result
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


if process_button:
    st.session_state.processing_complete = False
    st.session_state.issue_result = None
    process_document()


if st.session_state.processing_complete and st.session_state.issue_result:
    result = st.session_state.issue_result
    
    # Show success message
    st.success(f"Analysis completed in {st.session_state.processing_time:.2f} seconds")
    
    # Handle error case
    if "error" in result:
        st.error(f"Error during processing: {result['error']}")
        if "metadata" in result:
            st.json(result["metadata"])
        st.stop()
    
    # Get issues from result
    issues = []
    
    # Check different locations for issues based on how they were processed
    if "issues" in result:
        # Directly from specialized handler
        issues = result["issues"]
    elif "passes" in result and "issue_identification" in result["passes"]:
        # From standard orchestrator with passes
        issue_results = result["passes"]["issue_identification"]
        if "result" in issue_results and "issues" in issue_results["result"]:
            issues = issue_results["result"]["issues"]
    
    # Create tabs for different parts of the results
    issues_tab, metadata_tab = st.tabs(["Issues", "Metadata"])
    
    with issues_tab:
        # Display issues summary
        st.subheader(f"{len(issues)} Issues Identified")
        
        # Group issues by severity
        issues_by_severity = {}
        for issue in issues:
            severity = issue.get("severity", "medium").lower()
            if severity not in issues_by_severity:
                issues_by_severity[severity] = []
            issues_by_severity[severity].append(issue)
        
        # Show summary metrics
        cols = st.columns(4)
        cols[0].metric("Critical Issues", len(issues_by_severity.get("critical", [])), 
                      delta=None, delta_color="inverse")
        cols[1].metric("High Issues", len(issues_by_severity.get("high", [])),
                      delta=None, delta_color="inverse")
        cols[2].metric("Medium Issues", len(issues_by_severity.get("medium", [])),
                      delta=None, delta_color="inverse")
        cols[3].metric("Low Issues", len(issues_by_severity.get("low", [])),
                      delta=None, delta_color="inverse")
        
        # Display issues by severity
        if issues:
            # Define severity colors for visual cues
            severity_colors = {
                "critical": "❗❗ CRITICAL",
                "high": "❗ HIGH",
                "medium": "⚠️ MEDIUM",
                "low": "ℹ️ LOW"
            }
            
            # Display issues grouped by severity
            for severity in ["critical", "high", "medium", "low"]:
                if severity in issues_by_severity and issues_by_severity[severity]:
                    severity_issues = issues_by_severity[severity]
                    
                    with st.expander(f"{severity_colors.get(severity, severity.upper())} Issues ({len(severity_issues)})", 
                                    expanded=(severity in ["critical", "high"])):
                        for i, issue in enumerate(severity_issues):
                            title = issue.get("title", f"Issue {i+1}")
                            description = issue.get("description", "No description provided.")
                            speaker = issue.get("speaker", "")
                            context = issue.get("context", "")
                            
                            st.markdown(f"### {title}")
                            st.markdown(description)
                            
                            # Show metadata in columns
                            if speaker or (context and include_context):
                                meta_cols = st.columns(2)
                                if speaker:
                                    meta_cols[0].markdown(f"**Mentioned by:** {speaker}")
                                
                                if context and include_context:
                                    meta_cols[1].markdown(f"**Context:**")
                                    meta_cols[1].markdown(f"> {context}")
                            
                            # Add separator between issues
                            if i < len(severity_issues) - 1:
                                st.markdown("---")
        else:
            st.info("No issues were identified in this document.")
            
        # Export button for issues
        if issues:
            # Create formatted markdown for export
            export_md = "# Issues Identified\n\n"
            export_md += f"*Analysis completed on {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
            
            for severity in ["critical", "high", "medium", "low"]:
                if severity in issues_by_severity:
                    export_md += f"## {severity.upper()} Issues\n\n"
                    for issue in issues_by_severity[severity]:
                        title = issue.get("title", "Untitled Issue")
                        description = issue.get("description", "No description provided.")
                        speaker = issue.get("speaker", "")
                        context = issue.get("context", "")
                        
                        export_md += f"### {title}\n\n{description}\n\n"
                        if speaker:
                            export_md += f"**Mentioned by:** {speaker}\n\n"
                        if context:
                            export_md += f"**Context:** \"{context}\"\n\n"
            
            st.download_button(
                "Download Issues Report",
                export_md,
                file_name="issue_analysis.md",
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
    st.info("Upload a document to begin issue identification.")
    with st.expander("Types of Issues We Can Identify"):
        st.markdown("""
        - **Operational Challenges**: Inefficiencies, resource constraints
        - **Technical Issues**: System limitations, compatibility problems
        - **Strategic Concerns**: Market risks, competitive threats
        - **Organizational Problems**: Communication barriers, inefficiencies
        - **Regulatory Issues**: Legal risks, policy violations
        - **Customer Issues**: Service gaps, satisfaction problems
        """)