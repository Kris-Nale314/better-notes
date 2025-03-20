"""
Action Items Page for Better Notes.
Extracts tasks, assignments, and commitments from documents.
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
    page_title="Action Items - Better Notes",
    page_icon="✅",
    layout="wide"
)

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")

# Model selection
model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
selected_model = st.sidebar.selectbox("Model", model_options, index=0)

# Detail level
detail_options = ["essential", "detailed", "detailed-complex"]
detail_level = st.sidebar.radio(
    "Detail Level", 
    detail_options,
    format_func=lambda x: x.replace("-", " ").title(),
    help="Controls processing depth and result detail"
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
    help="Lower values give more deterministic outputs"
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
st.title("Action Item Extraction")
st.markdown("""
Upload a document to extract action items, tasks, commitments, and follow-up items.
The system will analyze the content and identify items that require action.
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
st.subheader("Extraction Focus (Optional)")
user_instructions = st.text_area(
    "Add specific focus for action item extraction:",
    placeholder="E.g., 'Focus on technical tasks', 'Look for deadlines and commitments', 'Identify ownership'",
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
        # Create custom instruction based on detail level
        detail_instruction = ""
        if detail_level == "essential":
            detail_instruction = "Focus only on clear, explicit action items with specific owners or deadlines."
        elif detail_level == "detailed-complex":
            detail_instruction = "Extract all possible action items, including implicit tasks and subtle commitments."
        else:  # detailed (default)
            detail_instruction = "Identify clear action items that represent tasks or commitments."
        
        # Combine with user instructions
        combined_instructions = f"{detail_instruction}\n{user_instructions if user_instructions else ''}"
            
        # Create processing options
        options = ProcessingOptions(
            model_name=selected_model,
            temperature=temperature,
            min_chunks=min_chunks,
            detail_level="detailed",  # Always detailed for action item extraction
            max_concurrent_chunks=max_concurrent if enable_parallel else 1,
            user_instructions=combined_instructions,
            include_metadata=True,
            # Important: Specify that we want to run the action_items pass
            passes=["action_items"]
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
        st.session_state.action_result = result
        st.session_state.processing_complete = True
        st.session_state.processing_time = time.time() - start_time
        
        # Clear progress indicators
        progress_container.empty()
        
        # Rerun to display results
        st.rerun()
        
    except Exception as e:
        progress_container.empty()
        st.error(f"Error extracting action items: {str(e)}")
        st.exception(e)

# Process the document when button is clicked
if process_button:
    st.session_state.processing_complete = False
    st.session_state.action_result = None
    process_document()

# Display results
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
    
    # Extract action items from the pass results
    pass_results = result.get("passes", {})
    action_results = pass_results.get("action_items", {})
    
    # Get the actions from the results structure
    actions = []
    if "result" in action_results and "actions" in action_results["result"]:
        actions = action_results["result"]["actions"]
    elif "actions" in action_results:
        actions = action_results["actions"]
        
    # Display the action items
    if actions:
        st.subheader(f"{len(actions)} Action Items Extracted")
        
        # Create a table of action items
        action_data = []
        for action in actions:
            task = action.get("task", "Untitled Task")
            owner = action.get("owner", "")
            deadline = action.get("deadline", "")
            description = action.get("description", "")
            
            action_data.append({
                "Task": task,
                "Owner": owner,
                "Deadline": deadline,
                "Description": description
            })
        
        # Display as a table with expanders for details
        for i, action in enumerate(action_data):
            # Create a unique key for each expander
            key = f"action_{i}"
            
            # Create a nicer display of the task with owner and deadline
            display = f"**{action['Task']}**"
            if action['Owner']:
                display += f" (Owner: {action['Owner']})"
            if action['Deadline']:
                display += f" - Due: {action['Deadline']}"
            
            with st.expander(display):
                if action['Description']:
                    st.markdown(f"{action['Description']}")
                
        # Create a summary view
        with st.expander("Action Items Summary", expanded=True):
            # Show counts
            st.markdown(f"**Total Action Items:** {len(actions)}")
            
            # Count actions with owners and deadlines
            with_owners = sum(1 for a in action_data if a["Owner"])
            with_deadlines = sum(1 for a in action_data if a["Deadline"])
            
            cols = st.columns(3)
            cols[0].metric("Total Items", len(actions))
            cols[1].metric("With Owners", with_owners)
            cols[2].metric("With Deadlines", with_deadlines)
            
            # Export options
            if st.button("Export Action Items"):
                # Create markdown export
                markdown = "# Action Items\n\n"
                markdown += f"*Extracted on {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
                
                for action in action_data:
                    markdown += f"- **{action['Task']}**"
                    
                    if action['Owner']:
                        markdown += f" (Owner: {action['Owner']})"
                    
                    if action['Deadline']:
                        markdown += f" - Due: {action['Deadline']}"
                    
                    markdown += "\n"
                    
                    if action['Description']:
                        markdown += f"  {action['Description']}\n"
                    
                    markdown += "\n"
                
                # Create a download button
                st.download_button(
                    "Download Action Items",
                    markdown,
                    file_name="action_items.md",
                    mime="text/markdown"
                )
                
            # Create a simple task list view
            st.markdown("### Task List")
            for action in action_data:
                task_text = action['Task']
                if action['Owner'] or action['Deadline']:
                    details = []
                    if action['Owner']:
                        details.append(f"Owner: {action['Owner']}")
                    if action['Deadline']:
                        details.append(f"Due: {action['Deadline']}")
                    task_text += f" ({', '.join(details)})"
                
                st.checkbox(task_text, key=f"task_check_{hash(task_text)}")
    
    else:
        st.info("No action items were found in this document.")
        
        # Show summary from the document for context
        if "summary" in result:
            with st.expander("Document Summary"):
                st.markdown(result["summary"])
    
    # Add metadata tab
    with st.expander("Processing Details"):
        metadata = result.get("metadata", {})
        if metadata:
            cols = st.columns(3)
            with cols[0]:
                st.metric("Model", metadata.get("model", "Unknown"))
            with cols[1]:
                st.metric("Chunks", metadata.get("chunks_processed", 0))
            with cols[2]:
                st.metric("Processing Time", f"{metadata.get('processing_time_seconds', 0):.2f}s")

# When no document is loaded, show placeholder
if not document_text and not st.session_state.processing_complete:
    st.info("Upload a document to begin action item extraction.")
    
    # Example of what the system can extract
    with st.expander("Types of Action Items We Can Extract"):
        st.markdown("""
        Our system extracts various types of action items, including:
        
        - **Tasks and Assignments**: Specific work items assigned to individuals
        - **Commitments and Promises**: Things people have committed to doing
        - **Follow-up Items**: Areas requiring additional investigation or action
        - **Decisions That Require Action**: Decisions that imply tasks to be completed
        - **Time-bound Activities**: Activities with deadlines or timeframes
        
        The system can identify both explicit action items ("John will create the report by Friday") 
        and implied action items ("We need to review the budget next quarter").
        """)