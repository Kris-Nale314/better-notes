"""
Multi-Agent Analysis Page for Better Notes.
Provides agent-based document analysis with various crews.
"""

import os
import time
import streamlit as st
from pathlib import Path
import tempfile
import json
import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import the orchestrator
from orchestrator import OrchestratorFactory

# Import lean components
from lean.options import ProcessingOptions
from lean.async_openai_adapter import AsyncOpenAIAdapter

# Import UI enhancements
from ui_utils.ui_enhance import (
    apply_custom_css, 
    render_agent_card, 
    enhance_result_display,
    format_log_entries,
    display_chat_interface
)

# Configure page
st.set_page_config(
    page_title="Multi-Agent Analysis - Better Notes",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for styling
apply_custom_css()

# Set up output directory structure
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def ensure_output_dirs():
    """Ensure all output directories exist"""
    for analysis_type in ["issues", "actions", "insights"]:
        dir_path = OUTPUT_DIR / analysis_type
        dir_path.mkdir(exist_ok=True)
    
    # Directory for agent intermediates
    (OUTPUT_DIR / "intermediates").mkdir(exist_ok=True)

ensure_output_dirs()

# --- Sidebar Configuration ---
st.sidebar.header("Analysis Settings")

# Model selection
model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
selected_model = st.sidebar.selectbox("Language Model", model_options, index=0)

# Detail level
detail_level = st.sidebar.select_slider(
    "Detail Level",
    options=["Essential", "Standard", "Comprehensive"],
    value="Standard",
    help="Controls the depth and comprehensiveness of the analysis"
)

# Processing settings
st.sidebar.subheader("Processing Settings")

# Temperature
temperature = st.sidebar.slider(
    "Temperature", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.2,
    step=0.1,
    help="Lower values give more consistent outputs, higher values more creative"
)

# Chunk settings
min_chunks = st.sidebar.slider(
    "Minimum Chunks",
    min_value=1,
    max_value=10,
    value=3,
    help="Minimum number of sections to divide the document into"
)

max_chunk_size = st.sidebar.slider(
    "Max Chunk Size",
    min_value=500,
    max_value=3000,
    value=1500,
    step=100,
    help="Maximum size of each document chunk (smaller values help prevent API errors)"
)

# Advanced settings
with st.sidebar.expander("Advanced Settings"):
    max_rpm = st.slider(
        "Max Requests Per Minute",
        min_value=5,
        max_value=30,
        value=10,
        step=1,
        help="Controls API request rate (higher values = faster but may hit rate limits)"
    )
    
    show_agent_details = st.checkbox("Show Agent Interactions", value=False)

# --- Main Content ---
st.title("Multi-Agent Document Analysis")
st.markdown("Select an analysis type and upload a document to generate insights using specialized AI agent teams.")

# --- Analysis Type Selection ---
st.subheader("Select Analysis Type")

analysis_types = {
    "issues": {
        "title": "Issues Identification",
        "description": "Identify problems, challenges, and risks in your document",
        "icon": "⚠️"
    },
    "actions": {
        "title": "Action Items Extraction",
        "description": "Extract tasks, commitments, and follow-up items",
        "icon": "✅"
    },
    "insights": {
        "title": "Context Insights",
        "description": "Understand document type, tone, themes, and notable points",
        "icon": "💡"
    }
}

# Create clickable cards for analysis types
analysis_cols = st.columns(3)

# Store the selected analysis type in session state
if "selected_analysis" not in st.session_state:
    st.session_state.selected_analysis = None

# Function to set selected analysis
def select_analysis(analysis_type):
    st.session_state.selected_analysis = analysis_type
    st.rerun()

# Create selection cards
with analysis_cols[0]:
    st.markdown(f"""
    <div class="analysis-card issues-card">
        <h3>{analysis_types['issues']['icon']} {analysis_types['issues']['title']}</h3>
        <p>{analysis_types['issues']['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Select Issues Analysis", key="select_issues", type="secondary"):
        select_analysis("issues")

with analysis_cols[1]:
    st.markdown(f"""
    <div class="analysis-card actions-card">
        <h3>{analysis_types['actions']['icon']} {analysis_types['actions']['title']}</h3>
        <p>{analysis_types['actions']['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Select Action Items", key="select_actions", type="secondary"):
        select_analysis("actions")

with analysis_cols[2]:
    st.markdown(f"""
    <div class="analysis-card insights-card">
        <h3>{analysis_types['insights']['icon']} {analysis_types['insights']['title']}</h3>
        <p>{analysis_types['insights']['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Select Context Insights", key="select_insights", type="secondary"):
        select_analysis("insights")

# Function to save output to file
def save_output_to_file(content: str, analysis_type: str) -> str:
    """
    Save analysis output to the outputs folder with a timestamp.
    
    Args:
        content: Content to save
        analysis_type: Type of analysis
        
    Returns:
        File path where content was saved
    """
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename with analysis-specific directory
    output_dir = OUTPUT_DIR / analysis_type
    output_dir.mkdir(exist_ok=True)
    
    filename = f"{analysis_type}_{timestamp}.md"
    filepath = output_dir / filename
    
    # Write content to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    return str(filepath)

# Function to save agent intermediate outputs
def save_intermediate_output(content: str, analysis_type: str, agent_type: str) -> str:
    """
    Save intermediate agent output for debugging and inspection.
    
    Args:
        content: Content to save
        analysis_type: Type of analysis (issues, actions, etc.)
        agent_type: Type of agent (extractor, aggregator, etc.)
        
    Returns:
        File path where content was saved
    """
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename with intermediates directory
    output_dir = OUTPUT_DIR / "intermediates"
    output_dir.mkdir(exist_ok=True)
    
    filename = f"{analysis_type}_{agent_type}_{timestamp}.json"
    filepath = output_dir / filename
    
    # Write content to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    return str(filepath)

# Display configuration for selected analysis type
if st.session_state.selected_analysis:
    selected_type = st.session_state.selected_analysis
    
    st.divider()
    
    # Show the selected analysis type header
    st.header(f"{analysis_types[selected_type]['icon']} {analysis_types[selected_type]['title']}")
    
    # Document upload
    st.subheader("Upload Document")
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
        st.warning("⚠️ Very large document detected. The analysis may be split into smaller chunks to prevent API errors. For best results with large documents, consider splitting it into separate uploads.")
    
    # Custom instructions
    st.subheader("Custom Instructions (Optional)")
    user_instructions = st.text_area(
        "Add specific instructions for the analysis:",
        placeholder=f"E.g., 'Focus on technical issues', 'Prioritize strategic opportunities'",
        help="Your instructions will guide how the agents analyze the document."
    )
    
    # Process button
    process_button = st.button(
        f"Analyze with {analysis_types[selected_type]['title']} Agents", 
        disabled=not document_text,
        type="primary",
        use_container_width=True
    )
    
    # Initialize session state for processing
    if "agent_result" not in st.session_state:
        st.session_state.agent_result = None
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    if "processing_time" not in st.session_state:
        st.session_state.processing_time = 0
    if "agent_logs" not in st.session_state:
        st.session_state.agent_logs = []
    if "agent_statuses" not in st.session_state:
        st.session_state.agent_statuses = {
            "extractor": "waiting",
            "aggregator": "waiting",
            "evaluator": "waiting",
            "formatter": "waiting"
        }
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "document_info" not in st.session_state:
        st.session_state.document_info = None
    
    # Function to process document
    def process_document():
        """Process the document with agent crews and handle UI updates."""
        # Store original settings in session state if not already there
        if "original_settings" not in st.session_state:
            st.session_state.original_settings = {
                "detail_level": detail_level,
                "temperature": temperature,
                "min_chunks": min_chunks,
                "user_instructions": user_instructions
            }
        
        # Check if we're doing a reanalysis and apply settings
        current_detail = detail_level
        current_temp = temperature
        current_chunks = min_chunks
        current_instructions = user_instructions
        
        if "reanalysis_settings" in st.session_state:
            # Get settings from reanalysis
            reanalysis = st.session_state.reanalysis_settings
            current_detail = reanalysis["detail_level"]
            current_temp = reanalysis["temperature"]
            current_chunks = reanalysis["min_chunks"]
            current_instructions = reanalysis["user_instructions"]
            
            # Clear reanalysis settings for next run
            del st.session_state.reanalysis_settings
            
            # Log the reanalysis
            timestamp = time.strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] Reanalyzing with updated settings: Detail={current_detail}, Temp={current_temp}"
            st.session_state.agent_logs.append(log_entry)
        
        # Get API key
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            st.error("OpenAI API key not found! Please set the OPENAI_API_KEY environment variable.")
            return
    
        
        # Set up progress tracking
        progress_container = st.container()
        progress_bar = progress_container.progress(0)
        status_text = progress_container.empty()
        
        # Agent status displays
        agent_status_container = st.container()
        agent_displays = agent_status_container.columns(4)
        
        # Initialize agent status displays
        agent_status_components = {
            "extractor": agent_displays[0].empty(),
            "aggregator": agent_displays[1].empty(),
            "evaluator": agent_displays[2].empty(),
            "formatter": agent_displays[3].empty()
        }
        
        # Initial render of agent cards
        for agent_type, component in agent_status_components.items():
            render_agent_card(agent_type, "waiting", component)
        
        # Agent logs container
        if show_agent_details:
            log_container = st.container()
            log_box = log_container.empty()
            st.session_state.agent_logs = []
        
        start_time = time.time()
        
        # Progress callback
        def update_progress(progress, message):
            progress_bar.progress(progress)
            status_text.text(message)
            
            # Add to agent logs
            if show_agent_details:
                timestamp = time.strftime("%H:%M:%S")
                log_entry = f"[{timestamp}] {message}"
                st.session_state.agent_logs.append(log_entry)
                log_html = format_log_entries(st.session_state.agent_logs)
                log_box.markdown(log_html, unsafe_allow_html=True)
            
            # Update agent statuses based on progress messages
            if "Extraction" in message or "extraction" in message.lower():
                update_agent_status("extractor", "working", agent_status_components)
            elif "Aggregating" in message or "aggregation" in message.lower():
                update_agent_status("extractor", "complete", agent_status_components)
                update_agent_status("aggregator", "working", agent_status_components)
            elif "Evaluating" in message or "evaluation" in message.lower():
                update_agent_status("aggregator", "complete", agent_status_components)
                update_agent_status("evaluator", "working", agent_status_components)
            elif "Formatting" in message or "formatting" in message.lower():
                update_agent_status("evaluator", "complete", agent_status_components)
                update_agent_status("formatter", "working", agent_status_components)
            elif "complete" in message.lower():
                update_agent_status("formatter", "complete", agent_status_components)
        
        try:
            # Initialize LLM client
            llm_client = AsyncOpenAIAdapter(
                model=selected_model,
                temperature=temperature
            )
            
            # Store LLM client in session state for chat interface
            st.session_state.llm_client = llm_client
            
            # Map detail level to a standardized value
            detail_map = {
                "Essential": "essential",
                "Standard": "standard",
                "Comprehensive": "comprehensive"
            }
            
            # Create orchestrator using the factory
            orchestrator = OrchestratorFactory.create_orchestrator(
                model=selected_model,
                temperature=temperature,
                max_chunk_size=max_chunk_size,
                verbose=show_agent_details,
                max_rpm=max_rpm
            )
            
            # Update RPM setting in all agents
            orchestrator.update_max_rpm(max_rpm)
            
            # Analyze document for chat interface context
            update_progress(0.1, "Analyzing document...")
            analysis_result = asyncio.run(orchestrator.analyzer.analyze_preview(document_text))
            st.session_state.analysis_result = analysis_result
            st.session_state.document_info = analysis_result
            
            # Process options
            options = {
                "crews": [selected_type],
                "min_chunks": min_chunks,
                "max_chunk_size": max_chunk_size,
                "user_preferences": {
                    "user_instructions": user_instructions if user_instructions else None,
                    "detail_level": detail_map.get(detail_level, "standard")
                }
            }
            
            # Process document
            results = orchestrator.process_document(
                document_text,
                options=options,
                progress_callback=update_progress
            )
            
            # Store results
            st.session_state.agent_result = results.get(selected_type)
            st.session_state.processing_complete = True
            st.session_state.processing_time = time.time() - start_time
            
            # Save intermediate outputs if available
            if show_agent_details and hasattr(orchestrator.issues_crew, "crew"):
                for agent_type in ["extractor", "aggregator", "evaluator", "formatter"]:
                    try:
                        # Try to get output from tasks
                        task_outputs = orchestrator.issues_crew.crew.outputs
                        for task in task_outputs:
                            if agent_type in task.name.lower():
                                output_content = json.dumps(task.output, indent=2)
                                save_intermediate_output(output_content, selected_type, agent_type)
                    except:
                        pass  # Silently continue if task outputs not available
            
            # Clear progress indicators
            progress_container.empty()
            
            # Rerun to display results
            st.rerun()
            
        except Exception as e:
            progress_container.empty()
            st.error(f"Error processing document: {str(e)}")
            st.exception(e)
    
    # Helper function to update agent status
    def update_agent_status(agent_type, new_status, components):
        st.session_state.agent_statuses[agent_type] = new_status
        render_agent_card(agent_type, new_status, components[agent_type])

    # Process the document when button is clicked
    if process_button:
        st.session_state.processing_complete = False
        st.session_state.agent_result = None
        st.session_state.agent_statuses = {
            "extractor": "waiting",
            "aggregator": "waiting",
            "evaluator": "waiting",
            "formatter": "waiting"
        }
        process_document()

    # Display results
    if st.session_state.processing_complete and st.session_state.agent_result:
        result = st.session_state.agent_result
        
        # Handle CrewOutput object
        if hasattr(result, 'raw_output'):
            result_text = result.raw_output
        elif hasattr(result, 'result'):
            result_text = result.result
        elif isinstance(result, str):
            result_text = result
        else:
            result_text = str(result)
        
        # Save the result to a file
        saved_filepath = save_output_to_file(result_text, selected_type)
        
        # Show success message with filepath
        st.success(f"Analysis completed in {st.session_state.processing_time:.2f} seconds and saved to {saved_filepath}")
        
        # Create tabs for different views of the results
        result_tabs = st.tabs(["Report", "Chat with Document", "Adjust Analysis", "Agent Interactions", "Technical Details"])
                
        with result_tabs[0]:
            st.subheader(f"{analysis_types[selected_type]['title']} Results")
            
            # Use the enhanced result display from ui_utils
            enhanced_html = enhance_result_display(result_text, selected_type)
            st.markdown(enhanced_html, unsafe_allow_html=True)
            
            # Refinement options
            st.subheader("Refine Results")
            refine_cols = st.columns(3)
            
            with refine_cols[0]:
                if st.button("Add More Detail", use_container_width=True):
                    with st.spinner("Adding more detail..."):
                        # This would connect to the refiner
                        st.info("Refinement functionality coming soon!")
            
            with refine_cols[1]:
                if st.button("Make More Concise", use_container_width=True):
                    with st.spinner("Making more concise..."):
                        # This would connect to the refiner
                        st.info("Refinement functionality coming soon!")
            
            with refine_cols[2]:
                if st.button("Reformat", use_container_width=True):
                    with st.spinner("Reformatting..."):
                        # This would connect to the refiner
                        st.info("Refinement functionality coming soon!")
            
            # Download option
            st.download_button(
                f"Download {analysis_types[selected_type]['title']}",
                data=result_text,
                file_name=os.path.basename(saved_filepath),
                mime="text/markdown"
            )
        
        with result_tabs[1]:
            st.subheader(f"Chat about this {analysis_types[selected_type]['title']} Report")
            
            # Use the chat interface from ui_utils
            display_chat_interface(
                llm_client=st.session_state.llm_client,
                document_text=document_text,
                summary_text=result_text,
                document_info=st.session_state.document_info
            )
        
        with result_tabs[2]:
            st.subheader(f"Adjust Analysis Settings")
            
            # Create a form for reanalysis settings
            with st.form("reanalysis_form"):
                st.write("Adjust settings and provide new instructions to reanalyze the document")
                
                # Allow adjusting key parameters
                new_detail_level = st.select_slider(
                    "Detail Level",
                    options=["Essential", "Standard", "Comprehensive"],
                    value=detail_level,
                    help="Controls the depth of the analysis"
                )
                
                new_temperature = st.slider(
                    "Temperature", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=temperature,
                    step=0.1,
                    help="Higher values = more creative, lower = more consistent"
                )
                
                new_min_chunks = st.slider(
                    "Minimum Chunks",
                    min_value=1,
                    max_value=10,
                    value=min_chunks,
                    help="Minimum number of sections to divide document into"
                )
                
                # Expanded instructions area
                new_instructions = st.text_area(
                    "Analysis Instructions",
                    value=user_instructions,
                    height=150,
                    help="Based on the chat, what specific aspects would you like the analysis to focus on?"
                )
                
                # Add specific focus options based on analysis type
                if selected_type == "issues":
                    focus_options = st.multiselect(
                        "Focus Areas",
                        options=["Technical Issues", "Budget Concerns", "Timeline Risks", 
                                "Resource Constraints", "Quality Problems", "Process Issues"],
                        default=[],
                        help="Select specific areas to emphasize in the analysis"
                    )
                elif selected_type == "actions":
                    focus_options = st.multiselect(
                        "Focus Areas",
                        options=["Immediate Actions", "Ownership Assignment", "Deadlines", 
                                "Decision Points", "Follow-up Items", "Meeting Outcomes"],
                        default=[],
                        help="Select specific areas to emphasize in the analysis"
                    )
                else:  # insights
                    focus_options = st.multiselect(
                        "Focus Areas",
                        options=["Key Themes", "Participant Dynamics", "Meeting Outcomes", 
                                "Decision Process", "Notable Quotes", "Context Analysis"],
                        default=[],
                        help="Select specific areas to emphasize in the analysis"
                    )
                    
                # Add focus areas to instructions if selected
                if focus_options:
                    if new_instructions:
                        new_instructions += "\n\nFocus areas: " + ", ".join(focus_options)
                    else:
                        new_instructions = "Focus areas: " + ", ".join(focus_options)
                
                # Reanalysis button
                reanalyze_submitted = st.form_submit_button("Reanalyze Document", type="primary")
            
            if reanalyze_submitted:
                # Store new settings in session state
                st.session_state.reanalysis_settings = {
                    "detail_level": new_detail_level,
                    "temperature": new_temperature,
                    "min_chunks": new_min_chunks,
                    "user_instructions": new_instructions
                }
                
                # Reset processing flags
                st.session_state.processing_complete = False
                st.session_state.agent_result = None
                st.session_state.agent_statuses = {
                    "extractor": "waiting",
                    "aggregator": "waiting",
                    "evaluator": "waiting",
                    "formatter": "waiting"
                }
                
                # Show processing message
                st.info("Starting reanalysis with updated settings...")
                st.rerun()   # Rerun to trigger processing with new settings     
            
        
        with result_tabs[3]:
            st.subheader("Agent Interaction Log")
            if st.session_state.agent_logs:
                # Use the log formatter from ui_utils
                log_html = format_log_entries(st.session_state.agent_logs)
                st.markdown(log_html, unsafe_allow_html=True)
            else:
                st.info("Agent interaction logs are not available. Enable 'Show Agent Interactions' in settings to see detailed logs in your next analysis.")
        
        with result_tabs[4]:
            st.subheader("Technical Details")
            tech_details = {
                "model_used": selected_model,
                "temperature": temperature,
                "detail_level": detail_level,
                "chunks": min_chunks,
                "max_chunk_size": max_chunk_size,
                "max_rpm": max_rpm,
                "document_length": len(document_text),
                "processing_time_seconds": round(st.session_state.processing_time, 2),
                "analysis_type": selected_type,
                "output_file": saved_filepath,
                "user_preferences": {
                    "user_instructions": user_instructions if user_instructions else "None provided"
                }
            }
            
            st.json(tech_details)

    else:
        # When no analysis type is selected, show placeholder
        st.info("Select an analysis type above to begin.")