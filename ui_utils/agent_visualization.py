"""
Agent visualization module for Better Notes.
Provides components for displaying agent status and activity.
"""

import streamlit as st
from typing import Dict, Any, List, Optional, Union

def display_agent_cards(agent_states: Dict[str, str], show_description: bool = True):
    """
    Display agent status as cards.
    
    Args:
        agent_states: Dictionary of agent names and their statuses
        show_description: Whether to show agent role descriptions
    """
    agent_info = {
        "planner": {
            "title": "Planner Agent",
            "role": "Creates document-specific instructions"
        },
        "extractor": {
            "title": "Extractor Agent",
            "role": "Identifies patterns in document chunks"
        },
        "aggregator": {
            "title": "Aggregator Agent",
            "role": "Combines and deduplicates findings"
        },
        "evaluator": {
            "title": "Evaluator Agent",
            "role": "Assesses importance and impact"
        },
        "formatter": {
            "title": "Formatter Agent",
            "role": "Creates structured report"
        },
        "reviewer": {
            "title": "Reviewer Agent",
            "role": "Ensures quality and alignment"
        }
    }
    
    status_icons = {
        "waiting": "⏳",
        "working": "🔄",
        "complete": "✅",
        "error": "❌"
    }
    
    for agent, status in agent_states.items():
        if agent not in agent_info:
            continue
            
        info = agent_info[agent]
        icon = status_icons.get(status, "⏳")
        
        st.markdown(f"""
            <div class="agent-card {status}">
                <h4>{icon} {info['title']}</h4>
                {f"<p>{info['role']}</p>" if show_description else ""}
                <p><small>Status: {status.capitalize()}</small></p>
            </div>
        """, unsafe_allow_html=True)

def display_compact_agent_status(agent_states: Dict[str, str]):
    """
    Display a compact, horizontal view of agent status.
    
    Args:
        agent_states: Dictionary of agent names and their statuses
    """
    columns = st.columns(len(agent_states))
    
    agent_order = ["planner", "extractor", "aggregator", "evaluator", "formatter", "reviewer"]
    agent_labels = {
        "planner": "Planner",
        "extractor": "Extractor",
        "aggregator": "Aggregator",
        "evaluator": "Evaluator",
        "formatter": "Formatter",
        "reviewer": "Reviewer"
    }
    
    status_icons = {
        "waiting": "⏳",
        "working": "🔄",
        "complete": "✅",
        "error": "❌"
    }
    
    # Show agents in the correct order
    ordered_agents = [a for a in agent_order if a in agent_states]
    
    for i, agent in enumerate(ordered_agents):
        if i >= len(columns):
            break
            
        status = agent_states[agent]
        icon = status_icons.get(status, "⏳")
        label = agent_labels.get(agent, agent.capitalize())
        
        with columns[i]:
            st.markdown(f"""
                <div style="text-align: center;">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
                    <div>{label}</div>
                    <div><small>{status.capitalize()}</small></div>
                </div>
            """, unsafe_allow_html=True)

def display_agent_progress(agent_states: Dict[str, str]):
    """
    Display a step progress indicator for agent pipeline.
    
    Args:
        agent_states: Dictionary of agent names and their statuses
    """
    agent_order = ["planner", "extractor", "aggregator", "evaluator", "formatter", "reviewer"]
    agent_labels = {
        "planner": "Planning",
        "extractor": "Extraction",
        "aggregator": "Aggregation",
        "evaluator": "Evaluation",
        "formatter": "Formatting",
        "reviewer": "Review"
    }
    
    status_icons = {
        "waiting": "⏳",
        "working": "🔄",
        "complete": "✓",
        "error": "✗"
    }
    
    # Calculate progress percentage
    ordered_agents = [a for a in agent_order if a in agent_states]
    progress_value = 0
    
    for i, agent in enumerate(ordered_agents):
        status = agent_states[agent]
        if status == "complete":
            progress_value = (i + 1) / len(ordered_agents)
        elif status == "working":
            progress_value = (i + 0.5) / len(ordered_agents)
            break
    
    # Render progress bar and steps
    st.markdown("""
        <div class="step-progress-container">
            <div class="step-progress-bar">
                <div class="step-progress-fill" style="width: {}%;"></div>
            </div>
    """.format(progress_value * 100), unsafe_allow_html=True)
    
    # Display each step
    for i, agent in enumerate(ordered_agents):
        status = agent_states[agent]
        icon = status_icons.get(status, "⏳")
        label = agent_labels.get(agent, agent.capitalize())
        
        if status == "complete":
            step_class = "complete"
        elif status == "working":
            step_class = "active"
        else:
            step_class = ""
        
        st.markdown(f"""
            <div class="step-item" style="width: {100/len(ordered_agents)}%;">
                <div class="step-icon {step_class}">{icon}</div>
                <div class="step-label">{label}</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def update_agent_status(
    agent_type: str, 
    new_status: str, 
    status_container=None,
    agent_states: Optional[Dict[str, str]] = None
):
    """
    Update the status of an agent and refresh the display.
    
    Args:
        agent_type: Type of agent to update
        new_status: New status value
        status_container: Optional container to update
        agent_states: Optional agent states dictionary to update
    """
    # Update session state if agent_states not provided
    if agent_states is None:
        if "agent_statuses" not in st.session_state:
            st.session_state.agent_statuses = {}
        
        st.session_state.agent_statuses[agent_type] = new_status
        agent_states = st.session_state.agent_statuses
    else:
        agent_states[agent_type] = new_status
    
    # Update the display if container provided
    if status_container:
        with status_container:
            status_container.empty()
            display_agent_progress(agent_states)

def create_agent_status_tracker():
    """
    Create a container for tracking agent status with update functionality.
    
    Returns:
        Tuple of (container, update_function)
    """
    status_container = st.container()
    
    # Initialize agent statuses if not already in session state
    if "agent_statuses" not in st.session_state:
        st.session_state.agent_statuses = {
            "planner": "waiting",
            "extractor": "waiting",
            "aggregator": "waiting",
            "evaluator": "waiting",
            "formatter": "waiting",
            "reviewer": "waiting"
        }
    
    # Display initial status
    with status_container:
        display_agent_progress(st.session_state.agent_statuses)
    
    # Create an update function that will update the specific container
    def update_status(agent_type, new_status):
        update_agent_status(agent_type, new_status, status_container)
    
    return status_container, update_status