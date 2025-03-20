"""
Action Items Page for Better Notes.
Extracts tasks, assignments, and follow-up items from documents.
"""

import os
import time
import tempfile
import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta

# Import lean components
from lean.options import ProcessingOptions
from lean.orchestrator import SummarizerFactory
from lean.async_openai_adapter import AsyncOpenAIAdapter
from passes.actions import ActionItemExtractor

# Import specialized action item handler
from passes.actions import create_action_extractor

# Configure page
st.set_page_config(
    page_title="Action Items - Better Notes",
    page_icon="✅",
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
    value=5,  # Default higher for action items as they're often scattered
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

# --- Action Items-Specific Options ---
with st.sidebar.expander("Action Items Options", expanded=True):
    filter_by_owner = st.checkbox(
        "Group by Owner", 
        value=True, 
        help="Group action items by the person responsible"
    )
    
    show_timeline = st.checkbox(
        "Show Timeline", 
        value=True, 
        help="Group action items by deadline timeframe"
    )
    
    include_description = st.checkbox(
        "Include Descriptions", 
        value=True, 
        help="Show detailed descriptions for each action item"
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
enable_caching = st.sidebar.checkbox("Enable Caching", value=True)

# --- Main Content ---
st.title("Action Item Extraction")
st.markdown("""
Upload a document to extract tasks, assignments, and follow-up items.
The system will identify who is responsible for what and when it's due.
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
st.subheader("Extraction Focus (Optional)")
user_instructions = st.text_area(
    "Add specific focus areas for action item extraction:",
    placeholder="E.g., 'Focus on engineering tasks', 'Extract items assigned to the marketing team'",
    help="Your instructions will guide what types of action items are extracted."
)

# Process button
process_button = st.button(
    "Extract Action Items", 
    disabled=not document_text,
    type="primary",
    use_container_width=True
)

# Initialize session state
if "action_result" not in st.session_state:
    st.session_state.action_result = None
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "processing_time" not in st.session_state:
    st.session_state.processing_time = 0

def process_document():
    """Process document and handle UI updates."""
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
        # Create pass_options dictionary
        pass_options = {
            "action_items": {
                "include_description": include_description,
                "filter_by_owner": filter_by_owner,
                "show_timeline": show_timeline
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
            passes=["action_items"],
            pass_options=pass_options
        )
        
        # Progress callback
        def update_progress(progress, message):
            progress_bar.progress(progress)
            status_text.text(message)
        
        # APPROACH 1: Use specialized action item extractor
        # Create LLM client
        llm_client = AsyncOpenAIAdapter(
            model=selected_model,
            api_key=api_key,
            temperature=temperature
        )
        
        # Create and use action extractor
        action_extractor = create_action_extractor(llm_client, options)
        
        # Basic document info
        document_info = {
            "is_meeting_transcript": _is_likely_transcript(document_text),
            "preview_analysis": {
                "key_topics": []  # Would be filled by analyzer in real pipeline
            }
        }
        
        result = action_extractor.process_document_sync(
            document_text,
            document_info=document_info,
            progress_callback=update_progress
        )
        
        """
        # APPROACH 2: Use standard orchestrator with passes
        pipeline = SummarizerFactory.create_pipeline(api_key=api_key, options=options)
        orchestrator = pipeline['orchestrator']
        
        # Process document
        result = orchestrator.process_document_sync(
            document_text,
            progress_callback=update_progress
        )
        
        # Reshape result for consistent handling
        if "passes" in result and "action_items" in result["passes"]:
            action_results = result["passes"]["action_items"]
            if "result" in action_results:
                # Keep the original result for reference
                result["original_result"] = result.copy()
                # Extract the action data to top level for easier access
                if "actions" in action_results["result"]:
                    result["actions"] = action_results["result"]["actions"]
        """
        
        # Store results
        st.session_state.action_result = result
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


def _is_likely_transcript(text: str) -> bool:
    """Simple detection of transcript-like documents."""
    import re
    
    # Check for common transcript patterns
    patterns = [
        r'\b\w+:',  # Name followed by colon
        r'\d{1,2}:\d{2}',  # Time stamp
        r'Speaker \d+:',  # Speaker notation
        r'\[.*?\]',  # Square bracket annotations
    ]
    
    # Check first 1000 chars for these patterns
    sample = text[:1000]
    for pattern in patterns:
        if len(re.findall(pattern, sample)) > 3:
            return True
    
    return False


if process_button:
    st.session_state.processing_complete = False
    st.session_state.action_result = None
    process_document()


if st.session_state.processing_complete and st.session_state.action_result:
    result = st.session_state.action_result
    
    # Show success message
    st.success(f"Analysis completed in {st.session_state.processing_time:.2f} seconds")
    
    # Handle error case
    if "error" in result:
        st.error(f"Error during processing: {result['error']}")
        if "metadata" in result:
            st.json(result["metadata"])
        st.stop()
    
    # Get action items
    actions = result.get("actions", [])
    actions_by_owner = result.get("actions_by_owner", {})
    unassigned_actions = result.get("unassigned_actions", [])
    timeline = result.get("timeline", {})
    
    # Create tabs for different views
    action_tab, timeline_tab, metadata_tab = st.tabs(["Actions", "Timeline", "Metadata"])
    
    with action_tab:
        # Display action items summary
        st.subheader(f"{len(actions)} Action Items Extracted")
        
        # Show metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Actions", len(actions))
        col2.metric("Assigned", result.get("total_assigned", len(actions) - len(unassigned_actions)))
        col3.metric("Unassigned", result.get("total_unassigned", len(unassigned_actions)))
        
        if filter_by_owner and actions_by_owner:
            # Display by owner
            st.markdown("## Actions by Owner")
            
            # Create owner expandable sections
            for owner, owner_actions in actions_by_owner.items():
                with st.expander(f"{owner} ({len(owner_actions)} actions)", expanded=True):
                    for action in owner_actions:
                        task = action.get("task", "")
                        deadline = action.get("deadline", "")
                        description = action.get("description", "")
                        
                        action_container = st.container()
                        with action_container:
                            task_row = st.columns([3, 1])
                            
                            # Display task with deadline
                            if deadline:
                                task_row[0].markdown(f"**{task}** (Due: {deadline})")
                            else:
                                task_row[0].markdown(f"**{task}**")
                                
                            # Add description if enabled
                            if include_description and description:
                                st.markdown(description)
                            
                            st.markdown("---")
            
            # Show unassigned actions
            if unassigned_actions:
                with st.expander(f"Unassigned Actions ({len(unassigned_actions)})", expanded=True):
                    for action in unassigned_actions:
                        task = action.get("task", "")
                        deadline = action.get("deadline", "")
                        description = action.get("description", "")
                        
                        action_container = st.container()
                        with action_container:
                            task_row = st.columns([3, 1])
                            
                            # Display task with deadline
                            if deadline:
                                task_row[0].markdown(f"**{task}** (Due: {deadline})")
                            else:
                                task_row[0].markdown(f"**{task}**")
                                
                            # Add description if enabled
                            if include_description and description:
                                st.markdown(description)
                            
                            st.markdown("---")
        else:
            # Display all actions in a list
            st.markdown("## All Actions")
            
            for action in actions:
                task = action.get("task", "")
                owner = action.get("owner", "")
                deadline = action.get("deadline", "")
                description = action.get("description", "")
                
                action_container = st.container()
                with action_container:
                    task_row = st.columns([3, 1])
                    
                    # Display task
                    task_text = f"**{task}**"
                    if owner:
                        task_text += f" (Owner: {owner})"
                    if deadline:
                        task_text += f" (Due: {deadline})"
                    
                    task_row[0].markdown(task_text)
                    
                    # Add description if enabled
                    if include_description and description:
                        st.markdown(description)
                    
                    st.markdown("---")
        
        # Export button
        if actions:
            # Create formatted markdown for export
            export_md = "# Action Items\n\n"
            export_md += f"*Extracted on {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
            
            if filter_by_owner and actions_by_owner:
                # Group by owner in export
                for owner, owner_actions in actions_by_owner.items():
                    export_md += f"## {owner}\n\n"
                    for action in owner_actions:
                        task = action.get("task", "")
                        deadline = action.get("deadline", "")
                        description = action.get("description", "")
                        
                        if deadline:
                            export_md += f"- **{task}** (Due: {deadline})\n"
                        else:
                            export_md += f"- **{task}**\n"
                            
                        if description:
                            export_md += f"  {description}\n"
                            
                        export_md += "\n"
                
                # Add unassigned actions
                if unassigned_actions:
                    export_md += "## Unassigned Actions\n\n"
                    for action in unassigned_actions:
                        task = action.get("task", "")
                        deadline = action.get("deadline", "")
                        description = action.get("description", "")
                        
                        if deadline:
                            export_md += f"- **{task}** (Due: {deadline})\n"
                        else:
                            export_md += f"- **{task}**\n"
                            
                        if description:
                            export_md += f"  {description}\n"
                            
                        export_md += "\n"
            else:
                # Simple list in export
                for action in actions:
                    task = action.get("task", "")
                    owner = action.get("owner", "")
                    deadline = action.get("deadline", "")
                    description = action.get("description", "")
                    
                    action_line = f"- **{task}**"
                    if owner:
                        action_line += f" (Owner: {owner})"
                    if deadline:
                        action_line += f" (Due: {deadline})"
                    
                    export_md += action_line + "\n"
                    
                    if description:
                        export_md += f"  {description}\n"
                        
                    export_md += "\n"
            
            st.download_button(
                "Download Action Items",
                export_md,
                file_name="action_items.md",
                mime="text/markdown"
            )
    
    with timeline_tab:
        if show_timeline and timeline:
            st.markdown("## Action Timeline")
            
            # Today's actions
            today_actions = timeline.get("today", [])
            if today_actions:
                with st.expander(f"Today ({len(today_actions)})", expanded=True):
                    for action in today_actions:
                        task = action.get("task", "")
                        owner = action.get("owner", "")
                        description = action.get("description", "")
                        
                        if owner:
                            st.markdown(f"**{task}** (Owner: {owner})")
                        else:
                            st.markdown(f"**{task}**")
                            
                        if include_description and description:
                            st.markdown(description)
                            
                        st.markdown("---")
            
            # This week's actions
            this_week_actions = timeline.get("this_week", [])
            if this_week_actions:
                with st.expander(f"This Week ({len(this_week_actions)})", expanded=True):
                    for action in this_week_actions:
                        task = action.get("task", "")
                        owner = action.get("owner", "")
                        deadline = action.get("deadline", "")
                        description = action.get("description", "")
                        
                        header = f"**{task}**"
                        if owner:
                            header += f" (Owner: {owner})"
                        if deadline:
                            header += f" (Due: {deadline})"
                            
                        st.markdown(header)
                            
                        if include_description and description:
                            st.markdown(description)
                            
                        st.markdown("---")
            
            # Next week's actions
            next_week_actions = timeline.get("next_week", [])
            if next_week_actions:
                with st.expander(f"Next Week ({len(next_week_actions)})", expanded=True):
                    for action in next_week_actions:
                        task = action.get("task", "")
                        owner = action.get("owner", "")
                        deadline = action.get("deadline", "")
                        description = action.get("description", "")
                        
                        header = f"**{task}**"
                        if owner:
                            header += f" (Owner: {owner})"
                        if deadline:
                            header += f" (Due: {deadline})"
                            
                        st.markdown(header)
                            
                        if include_description and description:
                            st.markdown(description)
                            
                        st.markdown("---")
            
            # This month's actions
            this_month_actions = timeline.get("this_month", [])
            if this_month_actions:
                with st.expander(f"This Month ({len(this_month_actions)})", expanded=True):
                    for action in this_month_actions:
                        task = action.get("task", "")
                        owner = action.get("owner", "")
                        deadline = action.get("deadline", "")
                        description = action.get("description", "")
                        
                        header = f"**{task}**"
                        if owner:
                            header += f" (Owner: {owner})"
                        if deadline:
                            header += f" (Due: {deadline})"
                            
                        st.markdown(header)
                            
                        if include_description and description:
                            st.markdown(description)
                            
                        st.markdown("---")
            
            # Future actions
            future_actions = timeline.get("future", [])
            if future_actions:
                with st.expander(f"Future ({len(future_actions)})", expanded=True):
                    for action in future_actions:
                        task = action.get("task", "")
                        owner = action.get("owner", "")
                        deadline = action.get("deadline", "")
                        description = action.get("description", "")
                        
                        header = f"**{task}**"
                        if owner:
                            header += f" (Owner: {owner})"
                        if deadline:
                            header += f" (Due: {deadline})"
                            
                        st.markdown(header)
                            
                        if include_description and description:
                            st.markdown(description)
                            
                        st.markdown("---")
            
            # No deadline
            no_deadline_actions = timeline.get("no_deadline", [])
            if no_deadline_actions:
                with st.expander(f"No Deadline ({len(no_deadline_actions)})", expanded=True):
                    for action in no_deadline_actions:
                        task = action.get("task", "")
                        owner = action.get("owner", "")
                        description = action.get("description", "")
                        
                        header = f"**{task}**"
                        if owner:
                            header += f" (Owner: {owner})"
                            
                        st.markdown(header)
                            
                        if include_description and description:
                            st.markdown(description)
                            
                        st.markdown("---")
        else:
            st.info("Timeline view is disabled or no timeline data is available.")
    
    with metadata_tab:
        # Display metadata
        st.subheader("Processing Details")
        metadata = result.get("metadata", {})
        if metadata:
            cols = st.columns(3)
            with cols[0]:
                st.metric("Model", metadata.get("model", selected_model))
            with cols[1]:
                st.metric("Chunks", metadata.get("chunks_processed", min_chunks))
            with cols[2]:
                st.metric("Processing Time", f"{st.session_state.processing_time:.2f}s")
            
            # Display other metadata
            filtered_metadata = {k: v for k, v in metadata.items() 
                              if k not in ["model", "chunks_processed", "processing_time_seconds"]}
            if filtered_metadata:
                st.json(filtered_metadata)
        
        # Display document info
        st.subheader("Document Information")
        doc_info = result.get("document_info", {})
        if doc_info:
            # Display transcript status
            is_transcript = doc_info.get("is_meeting_transcript", False)
            st.info(f"Document Type: {'Meeting Transcript' if is_transcript else 'Document'}")
            
            # Display additional document info
            filtered_doc_info = {k: v for k, v in doc_info.items() 
                              if k != "original_text_length" and not isinstance(v, (list, dict)) 
                              or (isinstance(v, (list, dict)) and len(str(v)) < 1000)}
            if filtered_doc_info:
                st.json(filtered_doc_info)

# When no document is loaded, show placeholder
if not document_text and not st.session_state.processing_complete:
    st.info("Upload a document to begin action item extraction.")
    with st.expander("Examples of Action Items We Can Extract"):
        st.markdown("""
        - **Explicit Assignments**: "John will prepare the report for next week's meeting"
        - **Personal Commitments**: "I'll follow up with the client by Friday"
        - **Group Tasks**: "The team needs to complete the security audit before the end of the month"
        - **Follow-up Items**: "We should revisit this topic at our next quarterly review"
        - **Implied Tasks**: "This issue must be fixed before the release" (unassigned action item)
        """)