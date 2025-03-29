"""
03_Issues_Test.py - Test script for the new LangChain Issues Analysis Pipeline.
Provides a simple Streamlit interface to test the new architecture.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
import tempfile

import streamlit as st

# Add parent directory to path to import from project root
sys.path.append(str(Path(__file__).parent.parent))

# Import the LangChain Issues Analyzer
from crews.langchain_issues_analysis import LangChainIssuesAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("issues_test")

# Configure page
st.set_page_config(
    page_title="Issues Analysis Test",
    page_icon="🧪",
    layout="wide"
)

# Title and description
st.title("🧪 Issues Analysis Test")
st.markdown("""
This page allows you to test the new LangChain Issues Analysis pipeline with improved agents.
Upload a document or paste text to analyze it for issues, problems, and risks.
""")

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
    help="Controls the depth of analysis"
)

# Focus areas
focus_options = ["Technical", "Process", "Resource", "Quality", "Risk", "Compliance"]
focus_areas = st.sidebar.multiselect(
    "Focus Areas",
    options=focus_options,
    default=[],
    help="Select specific types of issues to focus on"
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

# Document input
st.header("Document Input")
input_tabs = st.tabs(["Upload File", "Paste Text", "Sample Document"])

document_text = ""

with input_tabs[0]:
    uploaded_file = st.file_uploader("Upload a text document", type=["txt", "md"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            with open(tmp_path, "r", encoding="utf-8", errors="replace") as f:
                document_text = f.read()
            
            st.success(f"File loaded: {uploaded_file.name}")
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

with input_tabs[1]:
    pasted_text = st.text_area(
        "Paste document text here",
        height=300,
        placeholder="Paste your document text here..."
    )
    if pasted_text:
        document_text = pasted_text
        st.success(f"Text loaded: {len(document_text)} characters")

with input_tabs[2]:
    st.markdown("### Sample Document")
    st.markdown("Use this pre-loaded sample document to test the analysis pipeline.")
    
    sample_document = """
    Project Status Report: Database Migration Project
    
    Current Status: At Risk
    
    Executive Summary:
    The database migration project is currently behind schedule due to several technical challenges and resource constraints. The original timeline estimated completion by November 15, but current projections indicate a delay of at least 3 weeks.
    
    Key Issues:
    1. Technical Challenges: The legacy database structure has more inconsistencies than initially documented, requiring additional cleansing scripts.
    2. Resource Constraints: The DBA team is currently understaffed, with two key members being pulled into other critical projects.
    3. Integration Testing Failures: Initial testing revealed compatibility issues with three downstream systems that weren't identified in the planning phase.
    4. Budget Concerns: Additional licensing costs for the migration tools were not accounted for in the initial budget, creating a projected overage of $45,000.
    
    Mitigation Plans:
    - Requesting additional DBA resources from the enterprise pool
    - Developing workarounds for the integration issues with downstream systems
    - Evaluating alternative migration tools with lower licensing costs
    
    Next Steps:
    1. Meeting with steering committee to approve revised timeline
    2. Finalizing resource reallocation plan
    3. Completing comprehensive testing plan for downstream systems
    
    Please provide feedback on the proposed mitigation strategies by Friday.
    """
    
    st.text_area(
        "Sample document",
        sample_document,
        height=300,
        disabled=False  # Allow editing to customize the sample
    )
    
    if st.button("Use Sample Document"):
        document_text = sample_document
        st.success("Sample document loaded")

# Custom instructions
with st.expander("Custom Instructions (Optional)", expanded=False):
    user_instructions = st.text_area(
        "Add specific instructions for the analysis:",
        placeholder="E.g., 'Focus on technical issues', 'Prioritize security risks', 'Look for budget concerns'",
        help="Your instructions will guide how the agents analyze the document."
    )

# Process button
process_col1, process_col2 = st.columns([1, 1])
with process_col1:
    process_button = st.button(
        "Analyze Document", 
        disabled=not document_text,
        type="primary",
        use_container_width=True
    )

with process_col2:
    if debug_mode:
        architecture = st.radio(
            "Architecture",
            ["LangChain", "Original CrewAI (for comparison)"],
            horizontal=True,
            help="Select which implementation to use"
        )
    else:
        architecture = "LangChain"

# Function to display progress
def progress_callback(progress, message):
    progress_bar.progress(progress)
    status_text.text(message)
    
    # Add to log if debug mode is enabled
    if debug_mode:
        with st.session_state.log_lock:
            st.session_state.logs.append(f"[{progress:.1%}] {message}")
            log_text = "\n".join(st.session_state.logs[-20:])  # Keep last 20 logs
            debug_log.text_area("Processing Log", log_text, height=200)

# Initialize session state for logs
if 'logs' not in st.session_state:
    st.session_state.logs = []
    st.session_state.log_lock = False  # Simple lock

# Process document
if process_button:
    # Create progress containers
    st.header("Processing")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Debug log
    if debug_mode:
        debug_log = st.empty()
        st.session_state.log_lock = False
        st.session_state.logs = ["Starting analysis..."]
    
    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error("OpenAI API key not found! Please set the OPENAI_API_KEY environment variable.")
    else:
        try:
            # Prepare options
            options = {
                "detail_level": detail_level.lower(),
                "focus_areas": [area.lower() for area in focus_areas],
                "user_instructions": user_instructions,
                "min_chunks": num_chunks,
                "enable_reviewer": enable_reviewer
            }
            
            # Create analyzer
            start_time = time.time()
            
            if architecture == "LangChain":
                # Use our new LangChain implementation
                analyzer = LangChainIssuesAnalyzer(
                    api_key=api_key,
                    model=selected_model,
                    temperature=temperature,
                    verbose=debug_mode
                )
                
                # Process document
                result = analyzer.process_document_sync(
                    document_text,
                    options=options,
                    progress_callback=progress_callback
                )
            else:
                # Use the original CrewAI implementation for comparison
                from orchestrator_factory import OrchestratorFactory
                
                orchestrator = OrchestratorFactory.create_orchestrator(
                    api_key=api_key,
                    model=selected_model,
                    temperature=temperature,
                    verbose=debug_mode
                )
                
                # Process document
                result = orchestrator.process_document_sync(
                    document_text,
                    options=options,
                    progress_callback=progress_callback
                )
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Clear progress
            progress_bar.empty()
            status_text.empty()
            
            # Show results
            st.header("Analysis Results")
            st.success(f"Analysis completed in {execution_time:.2f} seconds")
            
            # Extract HTML content if available
            html_content = None
            if isinstance(result, dict) and "formatted_report" in result:
                html_content = result["formatted_report"]
            elif isinstance(result, str) and ("<div" in result or "<html" in result):
                html_content = result
            
            # Display the formatted report
            if html_content:
                st.components.v1.html(html_content, height=800, scrolling=True)
            else:
                # Fallback to displaying raw result
                st.json(result)
            
            # Show review information if available
            if isinstance(result, dict) and "review_result" in result:
                review = result["review_result"]
                st.subheader("Quality Assessment")
                
                if isinstance(review, dict):
                    # Extract assessment scores
                    assessment = review.get("assessment", {})
                    if assessment and isinstance(assessment, dict):
                        # Display metrics
                        metrics = [(k.replace("_score", "").replace("_", " ").title(), v) 
                                for k, v in assessment.items() 
                                if isinstance(v, (int, float))]
                        
                        if metrics:
                            metric_cols = st.columns(len(metrics))
                            for i, (key, value) in enumerate(metrics):
                                metric_cols[i].metric(key, f"{value}/5")
                    
                    # Show summary assessment
                    if "summary" in review:
                        st.info(review["summary"])
                    
                    # Show improvement suggestions
                    if "improvement_suggestions" in review and isinstance(review["improvement_suggestions"], list):
                        st.subheader("Suggested Improvements")
                        for suggestion in review["improvement_suggestions"]:
                            if isinstance(suggestion, dict):
                                st.markdown(f"- **{suggestion.get('area', 'General')}**: {suggestion.get('suggestion', '')}")
                            elif isinstance(suggestion, str):
                                st.markdown(f"- {suggestion}")
            
            # Show debug information
            if debug_mode:
                st.subheader("Debug Information")
                
                # Add tabs for different debug views
                debug_tabs = st.tabs(["Raw Result", "Metadata", "Plan"])
                
                with debug_tabs[0]:
                    st.json(result)
                
                with debug_tabs[1]:
                    if isinstance(result, dict) and "_metadata" in result:
                        st.json(result["_metadata"])
                    else:
                        st.warning("No metadata available")
                
                with debug_tabs[2]:
                    if isinstance(result, dict) and "_metadata" in result and "plan" in result["_metadata"]:
                        st.json(result["_metadata"]["plan"])
                    else:
                        st.warning("No plan available")
                
        except Exception as e:
            # Show error
            st.error(f"Error during analysis: {str(e)}")
            
            # Show stack trace in debug mode
            if debug_mode:
                st.expander("Error Details").code(traceback.format_exc())
            
            # Clear progress
            progress_bar.empty()
            status_text.empty()

# Instructions
with st.expander("How to Use This Test Page"):
    st.markdown("""
    ### Instructions
    
    1. **Input Document**: Upload a file, paste text, or use the sample document
    2. **Configure Settings**: Adjust settings in the sidebar
    3. **Analyze**: Click the "Analyze Document" button
    4. **Review Results**: Examine the analysis report
    
    ### Testing Notes
    
    - Try different detail levels to see how the analysis changes
    - Use focus areas to target specific types of issues
    - Compare the LangChain implementation with the original CrewAI implementation
    - Use debug mode to see raw outputs and processing logs
    
    ### Troubleshooting
    
    - If you encounter errors, check the API key is set correctly
    - Verify the configuration settings in your config files
    - Look for detailed error messages in the debug log
    """)

# Add information about the architecture
with st.sidebar.expander("About This Implementation"):
    st.markdown("""
    ### LangChain Implementation
    
    This test page uses a LangChain-based implementation of the issues analysis pipeline. Key features:
    
    - **Improved Architecture**: Uses the enhanced ProcessingContext, agents, and error handling
    - **LangChain Integration**: Leverages LangChain's LLM integration capabilities
    - **Pipeline Approach**: Maintains the same sequential processing stages as the original
    - **Agent Reuse**: Reuses your improved agent implementations with a LangChain adapter
    
    ### Components Used
    
    - **ProcessingContext**: Shared state container for the pipeline
    - **PlannerAgent**: Creates tailored instructions for other agents
    - **ExtractorAgent**: Finds comprehensive information from document chunks
    - **AggregatorAgent**: Combines and deduplicates extracted items
    - **EvaluatorAgent**: Assesses importance and organizes by priority
    - **FormatterAgent**: Creates well-structured reports for Streamlit
    - **ReviewerAgent**: Performs final quality assessment
    """)