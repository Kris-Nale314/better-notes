"""
Enhanced Orchestrator for Better Notes with integrated ProcessingContext.
Simplified implementation that works with the new agent architecture.
"""

import time
import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Union
import asyncio

from universal_llm_adapter import LLMAdapter
from config_manager import ConfigManager

logger = logging.getLogger(__name__)

class ProcessingContext:
    """
    Context object that flows through the agent pipeline.
    Provides standardized storage for document, chunks, results, and metadata.
    """
    
    def __init__(self, document_text: str, options: Optional[Dict[str, Any]] = None):
        """
        Initialize a processing context.
        
        Args:
            document_text: Document text to process
            options: Processing options
        """
        # Store document and options
        self.document_text = document_text
        self.options = options or {}
        
        # Document metadata
        self.document_info = {}
        
        # Chunking
        self.chunks = []  # List of document chunks
        self.chunk_metadata = []  # Metadata for each chunk
        
        # Processing results by stage
        self.results = {}
        
        # Agent instructions from planner
        self.agent_instructions = {}
        
        # Processing metadata
        self.metadata = {
            "start_time": time.time(),
            "run_id": f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "current_stage": None,
            "stages": {},
            "errors": []
        }
    
    def set_stage(self, stage_name: str) -> None:
        """
        Begin a processing stage.
        
        Args:
            stage_name: Name of the stage to start
        """
        self.metadata["current_stage"] = stage_name
        self.metadata["stages"][stage_name] = {
            "status": "started",
            "start_time": time.time()
        }
        logger.info(f"Starting stage: {stage_name}")
    
    def complete_stage(self, stage_name: str, result: Any = None) -> None:
        """
        Complete a processing stage.
        
        Args:
            stage_name: Name of the completed stage
            result: Result of the stage (optional)
        """
        if stage_name in self.metadata["stages"]:
            stage = self.metadata["stages"][stage_name]
            stage["status"] = "completed"
            stage["end_time"] = time.time()
            stage["duration"] = stage["end_time"] - stage["start_time"]
            
            if result is not None:
                self.results[stage_name] = result
            
            logger.info(f"Completed stage: {stage_name} in {stage['duration']:.2f}s")
    
    def fail_stage(self, stage_name: str, error: Union[str, Exception]) -> None:
        """
        Mark a stage as failed.
        
        Args:
            stage_name: Name of the failed stage
            error: Error message or exception
        """
        error_message = str(error)
        
        if stage_name in self.metadata["stages"]:
            stage = self.metadata["stages"][stage_name]
            stage["status"] = "failed"
            stage["end_time"] = time.time()
            stage["duration"] = stage["end_time"] - stage["start_time"]
            stage["error"] = error_message
            
            self.metadata["errors"].append({
                "stage": stage_name,
                "message": error_message,
                "time": time.time()
            })
            
            logger.error(f"Failed stage: {stage_name} - {error_message}")
    
    def get_processing_time(self) -> float:
        """
        Get the total processing time so far.
        
        Returns:
            Processing time in seconds
        """
        return time.time() - self.metadata["start_time"]
    
    def update_progress(self, progress: float, message: str, callback = None) -> None:
        """
        Update progress and call the progress callback if provided.
        
        Args:
            progress: Progress value (0.0 to 1.0)
            message: Progress message
            callback: Optional progress callback function
        """
        self.metadata["progress"] = progress
        self.metadata["progress_message"] = message
        
        # Call the callback if provided
        if callback:
            try:
                callback(progress, message)
            except Exception as e:
                logger.warning(f"Error in progress callback: {e}")
    
    def get_final_result(self) -> Dict[str, Any]:
        """
        Get the final processing result.
        
        Returns:
            Dictionary with processing results and metadata
        """
        # Get the formatted result (might be string for HTML or dict)
        formatted_result = self.results.get("formatting", {})
        
        # Create a base result dictionary
        if isinstance(formatted_result, str):
            # If formatted_result is a string (likely HTML), wrap it in a dictionary
            result = {"formatted_report": formatted_result}
        else:
            # Otherwise use it directly
            result = formatted_result
        
        # Add review result if available
        review_result = self.results.get("review")
        if review_result:
            result["review_result"] = review_result
        
        # Add metadata
        result["_metadata"] = {
            "run_id": self.metadata["run_id"],
            "processing_time": self.get_processing_time(),
            "document_info": self.document_info,
            "stages": self.metadata["stages"],
            "errors": self.metadata["errors"],
            "options": self.options
        }
        
        # Add plan if available
        if "planning" in self.results:
            result["_metadata"]["plan"] = self.results["planning"]
        
        return result


class Orchestrator:
    """
    Orchestrates the document processing workflow.
    Creates and manages the ProcessingContext through the pipeline.
    """
    
    def __init__(
        self, 
        llm_client = None,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.2,
        verbose: bool = True,
        max_chunk_size: int = 10000,
        max_rpm: int = 10,
        config_manager: Optional[ConfigManager] = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            llm_client: LLM client (will be wrapped in LLMAdapter)
            api_key: API key for LLM (if llm_client not provided)
            model: Model name
            temperature: Temperature for generation
            verbose: Whether to enable verbose mode
            max_chunk_size: Maximum chunk size
            max_rpm: Maximum requests per minute
            config_manager: Optional config manager
        """
        # Ensure we have a proper LLM adapter
        if llm_client is not None:
            # Wrap the existing client
            self.llm_client = LLMAdapter(
                llm_client=llm_client,
                model=model,
                temperature=temperature
            )
        else:
            # Create a new client
            self.llm_client = LLMAdapter(
                api_key=api_key,
                model=model,
                temperature=temperature
            )
        
        self.verbose = verbose
        self.max_chunk_size = max_chunk_size
        self.max_rpm = max_rpm
        self.config_manager = config_manager or ConfigManager()
        
        # Cache for crew instances
        self._crews = {}
        
        logger.info(f"Orchestrator initialized with model: {model}, max_rpm: {max_rpm}")
    
    async def process_document(
        self, 
        document_text: str,
        options: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Process a document through the appropriate pipeline.
        
        Args:
            document_text: Document text
            options: Processing options
            progress_callback: Progress callback
            
        Returns:
            Processing results
        """
        # Create a processing context
        context = ProcessingContext(document_text, options or {})
        
        try:
            # Determine crew type from options
            crew_type = "issues"  # Default crew type
            
            if options:
                if "crew_type" in options:
                    crew_type = options["crew_type"]
                elif "crews" in options and options["crews"]:
                    # Legacy support for 'crews' list
                    crew_type = options["crews"][0]
            
            logger.info(f"Processing document with crew type: {crew_type}")
            
            # Get or create the appropriate crew
            crew = self._get_crew(crew_type)
            
            # Process with the crew
            await crew.process_document_with_context(context, progress_callback)
            
            # Return the final result
            return context.get_final_result()
            
        except Exception as e:
            logger.error(f"Error in document processing: {e}")
            
            # Update context with error
            current_stage = context.metadata.get("current_stage")
            if current_stage:
                context.fail_stage(current_stage, str(e))
            
            # Return error result
            return {
                "error": str(e),
                "_metadata": {
                    "run_id": context.metadata["run_id"],
                    "processing_time": context.get_processing_time(),
                    "errors": context.metadata["errors"],
                    "error": True
                }
            }
    
    def _get_crew(self, crew_type: str):
        """
        Get or create a crew by type.
        
        Args:
            crew_type: Type of crew
            
        Returns:
            Crew instance
        """
        if crew_type not in self._crews:
            # Create the crew based on type
            if crew_type == "issues":
                try:
                    from crews.issues_crew import IssuesCrew
                    logger.info(f"Creating IssuesCrew instance")
                    
                    self._crews[crew_type] = IssuesCrew(
                        llm_client=self.llm_client,
                        verbose=self.verbose,
                        max_chunk_size=self.max_chunk_size,
                        max_rpm=self.max_rpm,
                        config_manager=self.config_manager
                    )
                except ImportError as e:
                    logger.error(f"Error importing IssuesCrew: {e}")
                    raise ValueError(f"Failed to import crew type: {crew_type}")
            else:
                raise ValueError(f"Unsupported crew type: {crew_type}")
        
        return self._crews[crew_type]