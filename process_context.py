"""
ProcessingContext: Enhanced context management for Better Notes.
Provides robust state tracking, error handling, progress monitoring, and pipeline visibility.
"""

import time
import logging
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Union
import asyncio
import traceback

logger = logging.getLogger(__name__)

class ProcessingContext:
    """
    Enhanced context object for document processing pipelines.
    Manages state, metadata, and results across processing stages with improved monitoring.
    """
    
    def __init__(self, document_text: str, options: Optional[Dict[str, Any]] = None):
        """Initialize processing context with document and options."""
        # Core content
        self.document_text = document_text
        self.options = options or {}
        
        # Document metadata (populated during analysis)
        self.document_info = {}
        
        # Chunking information
        self.chunks = []
        self.chunk_metadata = []
        
        # Results storage by stage
        self.results = {}
        
        # Agent instructions (from planner)
        self.agent_instructions = {}
        
        # Create unique run ID
        self.run_id = f"run-{uuid.uuid4().hex[:8]}-{int(time.time())}"
        
        # Processing metadata
        self.metadata = {
            "start_time": time.time(),
            "run_id": self.run_id,
            "current_stage": None,
            "stages": {},
            "errors": [],
            "warnings": [],
            "progress": 0.0,
            "progress_message": "Initializing",
            "options": self.options.copy()  # Store original options
        }
        
        # Progress callback
        self.progress_callback = None
        
        logger.info(f"Created ProcessingContext with run_id: {self.run_id}")
    
    def set_stage(self, stage_name: str) -> None:
        """Begin a processing stage with improved tracking."""
        prev_stage = self.metadata.get("current_stage")
        if prev_stage:
            logger.info(f"Transitioning from stage '{prev_stage}' to '{stage_name}'")
        else:
            logger.info(f"Starting first stage: '{stage_name}'")
        
        # Update metadata
        self.metadata["current_stage"] = stage_name
        self.metadata["stages"][stage_name] = {
            "status": "running",
            "start_time": time.time(),
            "progress": 0.0
        }
        
        # Update progress based on stage transition
        stage_progress = self.get_stage_progress()
        if stage_name in stage_progress:
            progress_value = stage_progress[stage_name]
            self.update_progress(progress_value, f"Starting {stage_name}")
    
    def complete_stage(self, stage_name: str, result: Any = None, had_error: bool = False) -> None:
        """Complete a processing stage and store result."""
        if stage_name not in self.metadata["stages"]:
            logger.warning(f"Completing unknown stage: {stage_name}")
            self.set_stage(stage_name)
            
        # Update stage metadata
        stage = self.metadata["stages"][stage_name]
        stage["status"] = "completed" if not had_error else "completed_with_errors"
        stage["end_time"] = time.time()
        stage["duration"] = stage["end_time"] - stage["start_time"]
        stage["progress"] = 1.0
        
        # Store validated result
        if result is not None:
            try:
                # Test serialization to ensure it can be stored
                json.dumps(result, default=str)
                self.results[stage_name] = result
                logger.info(f"Stage {stage_name} completed in {stage['duration']:.2f}s")
            except (TypeError, OverflowError) as e:
                logger.warning(f"Cannot serialize result for {stage_name}: {e}")
                # Store simplified version
                if isinstance(result, dict):
                    self.results[stage_name] = {k: str(v) for k, v in result.items()}
                else:
                    self.results[stage_name] = str(result)
                
                # Add warning
                self.add_warning(f"Result for stage {stage_name} was not fully serializable")
        
        # Update progress to completion of this stage
        next_stage_index = self._get_next_stage_index(stage_name)
        if next_stage_index is not None:
            # Use the starting progress of the next stage
            stages = list(self.get_stage_progress().keys())
            if next_stage_index < len(stages):
                next_stage = stages[next_stage_index]
                progress_value = self.get_stage_progress()[next_stage]
                self.update_progress(progress_value, f"Completed {stage_name}")
            else:
                # If this was the last stage, use 1.0
                self.update_progress(1.0, f"Completed {stage_name}")
        else:
            # Get progress values for all stages
            stage_progress = self.get_stage_progress()
            if stage_name in stage_progress:
                # If we can't find the next stage, use current + a small increment
                current_progress = stage_progress[stage_name]
                next_progress = min(current_progress + 0.05, 1.0)
                self.update_progress(next_progress, f"Completed {stage_name}")
    
    def _get_next_stage_index(self, stage_name: str) -> Optional[int]:
        """Get the index of the next stage after the given stage."""
        stages = list(self.get_stage_progress().keys())
        try:
            current_index = stages.index(stage_name)
            return current_index + 1
        except (ValueError, IndexError):
            return None
    
    def fail_stage(self, stage_name: str, error: Union[str, Exception]) -> None:
        """Mark a stage as failed with detailed error tracking."""
        error_message = str(error)
        logger.error(f"Stage {stage_name} failed: {error_message}")
        
        # Get traceback if available
        error_traceback = traceback.format_exc() if isinstance(error, Exception) else None
        
        # Update stage info
        if stage_name in self.metadata["stages"]:
            stage = self.metadata["stages"][stage_name]
            stage["status"] = "failed"
            stage["end_time"] = time.time()
            stage["duration"] = stage["end_time"] - stage["start_time"]
            stage["error"] = error_message
            if error_traceback:
                stage["traceback"] = error_traceback
        else:
            # Handle failure for stage that wasn't started
            self.metadata["stages"][stage_name] = {
                "status": "failed",
                "start_time": time.time(),
                "end_time": time.time(),
                "duration": 0,
                "error": error_message
            }
            if error_traceback:
                self.metadata["stages"][stage_name]["traceback"] = error_traceback
            
        # Add to errors list
        self.metadata["errors"].append({
            "stage": stage_name,
            "message": error_message,
            "time": time.time(),
            "traceback": error_traceback
        })
        
        # Update progress with error
        current_progress = self.metadata["progress"]
        self.update_progress(current_progress, f"Error in {stage_name}: {error_message}")
    
    def update_stage_progress(self, progress: float, message: Optional[str] = None) -> None:
        """Update progress for the current stage."""
        current_stage = self.metadata.get("current_stage")
        if not current_stage:
            return
            
        # Ensure progress is within bounds
        progress = max(0.0, min(1.0, progress))
        
        # Update stage progress
        stage = self.metadata["stages"][current_stage]
        stage["progress"] = progress
        
        if message:
            stage["message"] = message
        
        # Update overall progress based on stage weights
        self._update_overall_progress()
    
    def update_progress(self, progress: float, message: str, callback: Optional[Callable] = None) -> None:
        """Update overall progress and call the progress callback if provided."""
        # Ensure progress is within bounds
        progress = max(0.0, min(1.0, progress))
        
        # Update metadata
        self.metadata["progress"] = progress
        self.metadata["progress_message"] = message
        
        # Call the provided callback or stored callback
        actual_callback = callback or self.progress_callback
        if actual_callback:
            try:
                actual_callback(progress, message)
            except Exception as e:
                logger.warning(f"Error in progress callback: {e}")
    
    def _update_overall_progress(self) -> None:
        """Update overall progress based on stage progress and weights."""
        # Define stage weights (can be customized)
        stage_weights = {
            "document_analysis": 0.05,
            "chunking": 0.05,
            "planning": 0.10,
            "extraction": 0.35,
            "aggregation": 0.15,
            "evaluation": 0.10,
            "formatting": 0.10,
            "review": 0.10
        }
        
        total_weight = 0
        weighted_progress = 0
        
        for stage_name, stage_data in self.metadata["stages"].items():
            weight = stage_weights.get(stage_name, 0.1)
            total_weight += weight
            
            if stage_data["status"] == "completed":
                weighted_progress += weight
            elif stage_data["status"] == "running":
                weighted_progress += weight * stage_data.get("progress", 0)
        
        # Calculate overall progress
        if total_weight > 0:
            self.metadata["progress"] = weighted_progress / total_weight
    
    def add_warning(self, message: str) -> None:
        """Add a warning message to the context."""
        self.metadata["warnings"].append({
            "message": message,
            "time": time.time(),
            "stage": self.metadata.get("current_stage")
        })
        logger.warning(message)
    
    def add_substage_result(self, stage_name: str, substage_name: str, result: Any) -> None:
        """Add a result for a substage of processing."""
        if stage_name not in self.results:
            self.results[stage_name] = {}
            
        if not isinstance(self.results[stage_name], dict):
            # Convert existing result to dictionary with '_main' key
            self.results[stage_name] = {"_main": self.results[stage_name]}
            
        # Add substage result
        self.results[stage_name][substage_name] = result
    
    def get_processing_time(self) -> float:
        """Get total processing time so far."""
        return time.time() - self.metadata["start_time"]
    
    def get_final_result(self) -> Dict[str, Any]:
        """Create the final result dictionary with metadata."""
        # Get the formatted result
        formatted_result = self.results.get("formatting", {})
        
        # Create result dictionary
        if isinstance(formatted_result, str):
            result = {"formatted_report": formatted_result}
        else:
            result = formatted_result.copy() if isinstance(formatted_result, dict) else {}
        
        # Add review result if available
        review_result = self.results.get("review")
        if review_result:
            result["review_result"] = review_result
        
        # Add metadata
        result["_metadata"] = {
            "run_id": self.run_id,
            "processing_time": self.get_processing_time(),
            "document_info": self.document_info,
            "stages": self.metadata["stages"],
            "errors": self.metadata["errors"],
            "warnings": self.metadata["warnings"],
            "options": self.options
        }
        
        # Add plan if available
        if "planning" in self.results:
            result["_metadata"]["plan"] = self.results["planning"]
        
        return result
    
    def validate(self) -> bool:
        """Validate the context for consistency and completeness."""
        # Essential components check
        if not self.document_text:
            logger.error("Missing document_text in context")
            return False
        
        # Validate stage completion
        if "document_chunking" in self.metadata["stages"]:
            if not self.chunks:
                logger.error("Missing chunks after chunking stage")
                return False
        
        # Validate planning
        if "planning" in self.metadata["stages"]:
            if not self.agent_instructions:
                logger.warning("Missing agent_instructions after planning stage")
        
        # Check for errors
        if self.metadata["errors"]:
            logger.warning(f"Context contains {len(self.metadata['errors'])} errors")
            
        return True
    
    def track_substage_progress(self, 
                              stage_name: str, 
                              substage_name: str, 
                              progress: float, 
                              message: Optional[str] = None) -> None:
        """Track progress for a substage (like individual chunks in extraction)."""
        # Ensure stage exists
        if stage_name not in self.metadata["stages"]:
            return
            
        # Ensure substages container exists
        stage = self.metadata["stages"][stage_name]
        if "substages" not in stage:
            stage["substages"] = {}
            
        # Update substage
        if substage_name not in stage["substages"]:
            stage["substages"][substage_name] = {"progress": progress}
        else:
            stage["substages"][substage_name]["progress"] = progress
            
        if message:
            stage["substages"][substage_name]["message"] = message
        
        # Update overall stage progress based on substages
        substages = stage["substages"]
        if substages:
            # Average progress across all substages
            stage["progress"] = sum(s["progress"] for s in substages.values()) / len(substages)
            
        # Update overall progress
        self._update_overall_progress()
    
    def track_issue_count(self, stage_name: str, count: int) -> None:
        """Track the number of issues found at different stages."""
        if "issue_counts" not in self.metadata:
            self.metadata["issue_counts"] = {}
            
        self.metadata["issue_counts"][stage_name] = count
    
    # --- NEW ENHANCED METHODS ---
    
    def get_stage_progress(self) -> Dict[str, float]:
        """
        Get the starting progress value for each stage.
        Used for calculating progress during stage transitions.
        
        Returns:
            Dictionary mapping stage names to their starting progress values
        """
        return {
            "document_analysis": 0.0,
            "chunking": 0.05,
            "planning": 0.12,
            "extraction": 0.20,
            "aggregation": 0.55,
            "evaluation": 0.70,
            "formatting": 0.85,
            "review": 0.95
        }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get complete pipeline status for UI rendering.
        
        Returns:
            Dictionary with pipeline status information
        """
        return {
            "stages": self.metadata["stages"],
            "current_stage": self.metadata["current_stage"],
            "progress": self.metadata["progress"],
            "progress_message": self.metadata["progress_message"],
            "errors": self.metadata.get("errors", []),
            "warnings": self.metadata.get("warnings", []),
            "processing_time": self.get_processing_time(),
            "run_id": self.run_id
        }
    
    def mark_stage_skipped(self, stage_name: str, reason: str = "Skipped due to configuration") -> None:
        """
        Mark a stage as skipped (useful for disabled stages in config).
        
        Args:
            stage_name: Name of the stage to mark as skipped
            reason: Reason for skipping (default: "Skipped due to configuration")
        """
        if stage_name not in self.metadata["stages"]:
            self.metadata["stages"][stage_name] = {
                "status": "skipped",
                "start_time": time.time(),
                "end_time": time.time(),
                "duration": 0,
                "reason": reason
            }
        else:
            # Update existing stage
            stage = self.metadata["stages"][stage_name]
            stage["status"] = "skipped"
            stage["reason"] = reason
            if "end_time" not in stage:
                stage["end_time"] = time.time()
                stage["duration"] = stage["end_time"] - stage["start_time"]
        
        logger.info(f"Stage {stage_name} skipped: {reason}")
    
    def get_stage_status(self, stage_name: str) -> Dict[str, Any]:
        """
        Get the status of a specific stage.
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            Dictionary with stage status or empty dict if stage doesn't exist
        """
        return self.metadata["stages"].get(stage_name, {})
    
    def set_stage_metadata(self, stage_name: str, key: str, value: Any) -> None:
        """
        Set custom metadata for a stage.
        
        Args:
            stage_name: Name of the stage
            key: Metadata key
            value: Metadata value
        """
        if stage_name not in self.metadata["stages"]:
            logger.warning(f"Cannot set metadata for non-existent stage: {stage_name}")
            return
            
        if "metadata" not in self.metadata["stages"][stage_name]:
            self.metadata["stages"][stage_name]["metadata"] = {}
            
        self.metadata["stages"][stage_name]["metadata"][key] = value
    
    def reset_stage(self, stage_name: str) -> None:
        """
        Reset a stage to allow it to be rerun.
        
        Args:
            stage_name: Name of the stage to reset
        """
        if stage_name in self.metadata["stages"]:
            del self.metadata["stages"][stage_name]
        
        if stage_name in self.results:
            del self.results[stage_name]
            
        logger.info(f"Reset stage {stage_name} for potential rerun")
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report about the processing pipeline.
        Useful for debugging and performance analysis.
        
        Returns:
            Report dictionary with processing statistics
        """
        stage_stats = []
        for name, stage in self.metadata["stages"].items():
            stage_stats.append({
                "name": name,
                "status": stage.get("status", "unknown"),
                "duration": stage.get("duration", 0),
                "had_errors": "error" in stage
            })
        
        return {
            "run_id": self.run_id,
            "processing_time": self.get_processing_time(),
            "stages_completed": sum(1 for s in stage_stats if s["status"] == "completed"),
            "stages_failed": sum(1 for s in stage_stats if s["status"] == "failed"),
            "total_errors": len(self.metadata.get("errors", [])),
            "total_warnings": len(self.metadata.get("warnings", [])),
            "stage_stats": stage_stats,
            "options_summary": {k: v for k, v in self.options.items() if k not in ["api_key"]}
        }