"""
Progress tracking module for Better Notes.
Provides components for tracking and visualizing process progress.
"""

import streamlit as st
import time
from typing import Dict, Any, List, Optional, Callable, Union

class ProgressTracker:
    """
    Progress tracking utility for multi-stage processes.
    """
    
    def __init__(
        self, 
        stages: List[str], 
        container=None, 
        on_update: Optional[Callable] = None
    ):
        """
        Initialize a progress tracker.
        
        Args:
            stages: List of process stages in order
            container: Optional container to display progress in
            on_update: Optional callback when progress updates
        """
        self.stages = stages
        self.container = container or st.container()
        self.on_update = on_update
        
        # Initialize state (use only ONE method - I'm choosing the instance variable approach)
        self.state = {
            "current_stage": None,
            "stages": {stage: {"status": "waiting", "progress": 0.0, "start_time": None, "end_time": None}
                      for stage in stages},
            "overall_progress": 0.0,
            "start_time": None,
            "current_message": ""
        }
        
        # Display initial state
        self.render()
    
    def start(self):
        """Start tracking progress."""
        self.state["start_time"] = time.time()
        self.render()
    
    def update(
        self, 
        stage: str, 
        progress: float = None, 
        status: str = None, 
        message: str = None
    ):
        """
        Update progress for a stage.
        
        Args:
            stage: Stage to update
            progress: Progress value (0.0-1.0)
            status: Status string
            message: Progress message
        """
        if stage not in self.stages:
            return
            
        stage_state = self.state["stages"][stage]
        
        # Update stage status
        if status:
            stage_state["status"] = status
            
            # Handle start/end time based on status
            if status == "working" and not stage_state["start_time"]:
                stage_state["start_time"] = time.time()
                self.state["current_stage"] = stage
            elif status == "complete" and stage_state["start_time"] and not stage_state["end_time"]:
                stage_state["end_time"] = time.time()
                
                # Auto-start next stage if possible
                current_index = self.stages.index(stage)
                if current_index < len(self.stages) - 1:
                    next_stage = self.stages[current_index + 1]
                    self.state["stages"][next_stage]["status"] = "working"
                    self.state["stages"][next_stage]["start_time"] = time.time()
                    self.state["current_stage"] = next_stage
        
        # Update stage progress
        if progress is not None:
            stage_state["progress"] = max(0.0, min(1.0, progress))
        
        # Update overall progress
        self._update_overall_progress()
        
        # Update message
        if message:
            self.state["current_message"] = message
        
        # Call callback if provided
        if self.on_update:
            self.on_update(stage, stage_state["status"], self.state["overall_progress"])
        
        # Render updated state
        self.render()
    
    def _update_overall_progress(self):
        """Update overall progress based on stage progress."""
        stage_weights = {stage: 1 / len(self.stages) for stage in self.stages}
        
        # Calculate weighted progress
        overall = 0.0
        for stage, weight in stage_weights.items():
            stage_info = self.state["stages"][stage]
            
            if stage_info["status"] == "complete":
                # Completed stages contribute full weight
                stage_progress = 1.0
            elif stage_info["status"] == "working":
                # Working stages contribute partial weight based on progress
                stage_progress = stage_info["progress"]
            else:
                # Waiting stages contribute no progress
                stage_progress = 0.0
                
            overall += weight * stage_progress
        
        self.state["overall_progress"] = overall
    
    def render(self):
        """Render the current progress state."""
        with self.container:
            # Clear previous content
            self.container.empty()
            
            # Overall progress bar
            st.progress(self.state["overall_progress"])
            
            # Current message
            if self.state["current_message"]:
                st.caption(self.state["current_message"])
            
            # Display time elapsed if started
            if self.state["start_time"]:
                elapsed = time.time() - self.state["start_time"]
                st.caption(f"Time elapsed: {elapsed:.1f} seconds")
            
            # Render stage progress
            cols = st.columns(len(self.stages))
            
            for i, stage in enumerate(self.stages):
                stage_info = self.state["stages"][stage]
                
                with cols[i]:
                    # Determine icon based on status
                    if stage_info["status"] == "complete":
                        icon = "✅"
                        color = "#20c997"
                    elif stage_info["status"] == "working":
                        icon = "🔄"
                        color = "#ff9f43"
                    elif stage_info["status"] == "error":
                        icon = "❌"
                        color = "#ff5252"
                    else:
                        icon = "⏳"
                        color = "#6c757d"
                    
                    # Display stage status
                    st.markdown(f"""
                        <div style="text-align: center; margin-bottom: 10px;">
                            <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">{icon}</div>
                            <div style="font-weight: 500; color: {color};">{stage.capitalize()}</div>
                            <div style="font-size: 0.8rem;">
                                {stage_info["status"].capitalize()}
                                {f' ({stage_info["progress"]*100:.0f}%)' if stage_info["status"] == "working" else ""}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    

def display_progress_bar(
    progress: float,
    message: str = None,
    container = None
):
    """
    Display a simple progress bar with optional message.
    
    Args:
        progress: Progress value (0.0-1.0)
        message: Optional progress message
        container: Optional container to display in
    """
    container = container or st.empty()
    
    with container:
        st.progress(progress)
        if message:
            st.caption(message)

def create_progress_callback(container = None):
    """
    Create a progress callback function for updating progress.
    
    Args:
        container: Optional container to display progress in
        
    Returns:
        Callback function for updating progress
    """
    container = container or st.container()
    progress_bar = container.progress(0.0)
    message_container = container.empty()
    
    def update_progress(progress: float, message: str = None):
        progress_bar.progress(progress)
        if message:
            message_container.caption(message)
    
    return update_progress

def log_progress(
    message: str, 
    log_container = None, 
    max_logs: int = 10
):
    """
    Log a progress message to a container.
    
    Args:
        message: Message to log
        log_container: Optional container to display logs in
        max_logs: Maximum number of logs to display
    """
    # Initialize log history if needed
    if "progress_logs" not in st.session_state:
        st.session_state.progress_logs = []
    
    # Add timestamp to log
    timestamp = time.strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    
    # Add to log history
    st.session_state.progress_logs.append(log_entry)
    
    # Trim if needed
    if len(st.session_state.progress_logs) > max_logs:
        st.session_state.progress_logs = st.session_state.progress_logs[-max_logs:]
    
    # Display logs
    log_container = log_container or st.container()
    with log_container:
        log_container.empty()
        
        # Format logs
        log_html = []
        for entry in st.session_state.progress_logs:
            entry_class = "log-entry"
            if "error" in entry.lower() or "failed" in entry.lower():
                entry_class += " status-error"
            elif "complete" in entry.lower() or "success" in entry.lower():
                entry_class += " status-complete"
                
            log_html.append(f'<div class="{entry_class}">{entry}</div>')
        
        # Display logs
        log_container.markdown("\n".join(log_html), unsafe_allow_html=True)