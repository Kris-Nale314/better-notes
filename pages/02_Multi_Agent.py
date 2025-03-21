# pages/02_Multi_Agent.py
import os
import time
import streamlit as st
from pathlib import Path
import tempfile
import json

from lean.options import ProcessingOptions
from orchestrator import Orchestrator
from lean.async_openai_adapter import AsyncOpenAIAdapter

# Configure page
st.set_page_config(
    page_title="Multi-Agent Analysis - Better Notes",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS with darker, more subdued colors for dark theme
st.markdown("""
<style>
    .agent-card {
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0px;
        background-color: rgba(70, 70, 70, 0.2);
        border-left: 4px solid #4e8cff;
    }
    .agent-working {
        border-left: 4px solid #ff9f43;
        background-color: rgba(80, 70, 60, 0.2);
    }
    .agent-complete {
        border-left: 4px solid #20c997;
        background-color: rgba(60, 80, 60, 0.2);
    }
    .agent-card h4 {
        margin-top: 0;
    }
    .analysis-card {
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0px;
        transition: transform 0.2s, box-shadow 0.2s;
        background-color: rgba(60, 60, 70, 0.2);
    }
    .analysis-card:hover {
        transform: translateY(-2px);
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.2);
    }
    .issues-card {
        border-left: 4px solid #4e8cff;
    }
    .actions-card {
        border-left: 4px solid #ff9f43;
    }
    .insights-card {
        border-left: 4px solid #20c997;
    }
    .log-entry {
        padding: 5px 10px;
        margin: 5px 0;
        border-radius: 4px;
        background-color: rgba(60, 60, 70, 0.3);
        font-family: monospace;
    }
    .flow-diagram {
        background-color: rgba(60, 60, 80, 0.2);
        border-radius: 8px;
        padding: 15px;
    }
    .flow-box {
        padding: 8px;
        text-align: center;
        border-radius: 5px;
        margin-bottom: 15px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.title("Multi-Agent Document Analysis")

# Introduction with columns
intro_col1, intro_col2 = st.columns([3, 2])

with intro_col1:
    st.markdown("""
    ### Transform your documents with collaborative AI agents
    
    Our multi-agent approach divides analysis among specialized AI agents that collaborate to produce deeper insights from your documents.
    
    Each agent has its own expertise, working together like a team of analysts to extract more meaningful information.
    """)
    
    with st.expander("Learn how multi-agent analysis works"):
        st.markdown("""
        Our system uses a team of specialized agents:
        
        1. **Extractor Agents** analyze document chunks to identify specific elements
        2. **Aggregator Agent** combines and deduplicates the findings
        3. **Evaluator Agent** assesses importance, severity, or relevance
        4. **Formatter Agent** creates a clear, structured final report
        
        This approach allows for more thorough analysis through specialization, parallel processing of large documents, and higher quality through collaborative refinement.
        """)

with intro_col2:
    # Simple diagram showing agent collaboration with HTML
    st.markdown("""
    <div class="flow-diagram" style="text-align: center;">
        <h4>Agent Collaboration Flow</h4>
        <div style="margin: 10px auto; width: 180px;">
            <div class="flow-box" style="background-color: #4e8cff;">Document</div>
            <div style="border-left: 2px dashed #888; height: 15px; margin-left: 90px;"></div>
            <div class="flow-box" style="background-color: #6c5ce7;">Extractors</div>
            <div style="border-left: 2px solid #888; height: 15px; margin-left: 90px;"></div>
            <div class="flow-box" style="background-color: #00b894;">Aggregator</div>
            <div style="border-left: 2px solid #888; height: 15px; margin-left: 90px;"></div>
            <div class="flow-box" style="background-color: #fdcb6e;">Evaluator</div>
            <div style="border-left: 2px solid #888; height: 15px; margin-left: 90px;"></div>
            <div class="flow-box" style="background-color: #e17055;">Formatter</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Analysis Type Selection ---
st.header("Select Analysis Type")

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

# Display configuration for selected analysis type
if st.session_state.selected_analysis:
    selected_type = st.session_state.selected_analysis
    
    st.divider()
    
    # Show the selected analysis type header
    st.header(f"{analysis_types[selected_type]['icon']} {analysis_types[selected_type]['title']} Configuration")
    
    # Two-column layout for configuration
    config_col1, config_col2 = st.columns([3, 2])
    
    with config_col1:
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
                height=300,
                placeholder="Paste your document text here..."
            )
            if pasted_text:
                document_text = pasted_text
                st.success(f"Text loaded: {len(document_text)} characters")
        
        # Custom instructions
        st.subheader("Custom Instructions (Optional)")
        user_instructions = st.text_area(
            "Add specific instructions for the analysis:",
            placeholder=f"E.g., 'Focus on technical issues', 'Prioritize strategic opportunities'",
            help="Your instructions will guide how the agents analyze the document."
        )
    
    with config_col2:
        # Model selection
        st.subheader("Model Settings")
        model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        selected_model = st.selectbox("Language Model", model_options, index=0)
        
        # Temperature
        temperature = st.slider(
            "Temperature", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.4,
            step=0.1,
            help="Lower values give more deterministic outputs, higher values more creative"
        )
        
        # Agent verbosity
        st.subheader("Agent Visibility")
        show_agent_details = st.checkbox("Show Agent Interactions", value=True)
        
        # Number of chunks
        st.subheader("Processing Settings")
        min_chunks = st.slider(
            "Minimum Chunks",
            min_value=2,
            max_value=50,
            value=15,
            help="Minimum number of sections to divide the document into"
        )
        
        # Type-specific settings
        st.subheader("Analysis-Specific Settings")
        
        if selected_type == "issues":
            severity_filter = st.multiselect(
                "Include Severity Levels",
                ["Critical", "High", "Medium", "Low"],
                default=["Critical", "High", "Medium", "Low"]
            )
            
            group_by = st.radio(
                "Group Issues By",
                ["Severity", "Category", "None"],
                index=0
            )
            
        elif selected_type == "actions":
            commitment_threshold = st.select_slider(
                "Minimum Commitment Level",
                options=["Any mention", "Possible", "Probable", "Definite only"],
                value="Probable"
            )
            
            group_by = st.radio(
                "Group Action Items By",
                ["Owner", "Timeframe", "None"],
                index=0
            )
            
        elif selected_type == "insights":
            insight_depth = st.select_slider(
                "Analysis Depth",
                options=["Basic", "Standard", "Detailed"],
                value="Standard"
            )
            
            highlight_quotes = st.slider(
                "Number of Highlight Quotes",
                min_value=1,
                max_value=10,
                value=5
            )
    
    # Process button
    st.divider()
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
    
    # Function to process document
    def process_document():
        """Process the document with agent crews and handle UI updates."""
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
                log_display = "\n".join([f'<div class="log-entry">{entry}</div>' for entry in st.session_state.agent_logs])
                log_box.markdown(log_display, unsafe_allow_html=True)
            
            # Update agent statuses based on progress messages
            if "Extraction" in message:
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
            
            # Create orchestrator
            orchestrator = Orchestrator(llm_client, verbose=show_agent_details)
            
            # Process options
            options = {
                "crews": [selected_type],
                "min_chunks": min_chunks,
                "user_preferences": {
                    "user_instructions": user_instructions if user_instructions else None
                }
            }
            
            # Add type-specific settings
            if selected_type == "issues":
                options["user_preferences"]["severity_filter"] = [s.lower() for s in severity_filter]
                options["user_preferences"]["group_by"] = group_by.lower()
            elif selected_type == "actions":
                commitment_map = {
                    "Any mention": "any",
                    "Possible": "possible",
                    "Probable": "probable",
                    "Definite only": "definite"
                }
                options["user_preferences"]["commitment_threshold"] = commitment_map[commitment_threshold]
                options["user_preferences"]["group_by"] = group_by.lower()
            elif selected_type == "insights":
                depth_map = {
                    "Basic": "basic",
                    "Standard": "standard",
                    "Detailed": "detailed"
                }
                options["user_preferences"]["insight_depth"] = depth_map[insight_depth]
                options["user_preferences"]["highlight_quotes"] = highlight_quotes
            
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
            
            # Clear progress indicators
            progress_container.empty()
            
            # Rerun to display results
            st.rerun()
            
        except Exception as e:
            progress_container.empty()
            st.error(f"Error processing document: {str(e)}")
            st.exception(e)
    
    # Helper function to render agent cards
    def render_agent_card(agent_type, status, component):
        agent_info = {
            "extractor": {
                "title": "Extractor Agent", 
                "role": f"Identifying {selected_type}"
            },
            "aggregator": {
                "title": "Aggregator Agent", 
                "role": "Combining and deduplicating"
            },
            "evaluator": {
                "title": "Evaluator Agent", 
                "role": "Assessing importance and relevance"
            },
            "formatter": {
                "title": "Formatter Agent", 
                "role": "Creating structured report"
            }
        }
        
        status_icons = {
            "waiting": "⏳",
            "working": "🔄",
            "complete": "✅"
        }
        
        status_classes = {
            "waiting": "agent-card",
            "working": "agent-card agent-working",
            "complete": "agent-card agent-complete"
        }
        
        component.markdown(f"""
        <div class="{status_classes[status]}">
            <h4>{status_icons[status]} {agent_info[agent_type]['title']}</h4>
            <p>{agent_info[agent_type]['role']}</p>
            <p><small>Status: {status.capitalize()}</small></p>
        </div>
        """, unsafe_allow_html=True)
    
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
        
        # Show success message
        st.success(f"Analysis completed in {st.session_state.processing_time:.2f} seconds")
        
        # Handle CrewOutput object
        if hasattr(result, 'raw_output'):
            result_text = result.raw_output
        elif hasattr(result, 'result'):
            result_text = result.result
        elif isinstance(result, str):
            result_text = result
        else:
            result_text = str(result)
        
        # Create tabs for different views of the results
        result_tabs = st.tabs(["Report", "Agent Interactions", "Technical Details"])
        
        with result_tabs[0]:
            st.subheader(f"{analysis_types[selected_type]['title']} Results")
            st.markdown(result_text)
            
            # Download option
            st.download_button(
                f"Download {analysis_types[selected_type]['title']}",
                data=result_text,
                file_name=f"{selected_type}_analysis.md",
                mime="text/markdown"
            )
        
        with result_tabs[1]:
            st.subheader("Agent Interaction Log")
            if st.session_state.agent_logs:
                log_display = "\n".join([f'<div class="log-entry">{entry}</div>' for entry in st.session_state.agent_logs])
                st.markdown(log_display, unsafe_allow_html=True)
            else:
                st.info("Agent interaction logs are not available. Enable 'Show Agent Interactions' in settings to see detailed logs in your next analysis.")
        
        with result_tabs[2]:
            st.subheader("Technical Details")
            tech_details = {
                "model_used": selected_model,
                "temperature": temperature,
                "chunks": min_chunks,
                "document_length": len(document_text),
                "processing_time_seconds": round(st.session_state.processing_time, 2),
                "analysis_type": selected_type,
                "user_preferences": {}
            }
            
            # Add type-specific settings to technical details
            if selected_type == "issues":
                tech_details["user_preferences"]["severity_filter"] = severity_filter
                tech_details["user_preferences"]["group_by"] = group_by
            elif selected_type == "actions":
                tech_details["user_preferences"]["commitment_threshold"] = commitment_threshold
                tech_details["user_preferences"]["group_by"] = group_by
            elif selected_type == "insights":
                tech_details["user_preferences"]["insight_depth"] = insight_depth
                tech_details["user_preferences"]["highlight_quotes"] = highlight_quotes
            
            st.json(tech_details)

else:
    # When no analysis type is selected, show placeholder
    st.info("Select an analysis type above to begin.")