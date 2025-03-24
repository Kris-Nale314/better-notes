"""
Simplified Orchestrator - Coordinates the document processing pipeline.
Manages the flow of data through the multi-agent system with minimal complexity.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, Callable

# Import core components
from universal_llm_adapter import UniversalLLMAdapter
from config_manager import ConfigManager, ProcessingOptions

# Import processing components
from lean.document import DocumentAnalyzer
from lean.chunker import DocumentChunker

# Import agent
from agents.planner import PlannerAgent

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Unified orchestrator for document processing.
    Coordinates document analysis, chunking, and multi-agent processing.
    """
    
    def __init__(
        self,
        llm_client,
        analyzer=None,
        chunker=None,
        planner=None,
        verbose: bool = True,
        max_chunk_size: int = 10000,
        max_rpm: int = 10,
        config_manager: Optional[ConfigManager] = None
    ):
        """
        Initialize the orchestrator with all necessary components.
        
        Args:
            llm_client: Universal LLM adapter
            analyzer: Document analyzer (created if None)
            chunker: Document chunker (created if None)
            planner: Planner agent (created if None)
            verbose: Whether to enable verbose logging
            max_chunk_size: Maximum chunk size
            max_rpm: Maximum requests per minute
            config_manager: Config manager (created if None)
        """
        # Create adapter if needed
        if not isinstance(llm_client, UniversalLLMAdapter):
            self.llm_client = UniversalLLMAdapter(llm_client=llm_client)
        else:
            self.llm_client = llm_client
        
        self.verbose = verbose
        self.max_chunk_size = max_chunk_size
        self.max_rpm = max_rpm
        
        # Create config manager if needed
        self.config_manager = config_manager or ConfigManager()
        
        # Create components if not provided
        self.analyzer = analyzer or DocumentAnalyzer(self.llm_client)
        self.chunker = chunker or DocumentChunker()
        
        # Load planner config
        planner_config = self.config_manager.get_config("planner")
        
        # Create planner if needed
        self.planner = planner or PlannerAgent(
            llm_client=self.llm_client,
            config=planner_config,
            verbose=verbose,
            max_chunk_size=max_chunk_size,
            max_rpm=max_rpm
        )
        
        # For tracking current process
        self.run_id = None
        self.start_time = None
        
        # Cached crews
        self._crews = {}
        
        logger.info("Orchestrator initialized successfully")
    
    def process_document(
        self, 
        document_text: str,
        options: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process a document through the pipeline.
        
        Args:
            document_text: Document text to process
            options: Processing options
            progress_callback: Callback for progress updates
            
        Returns:
            Processing results
        """
        # Initialize run tracking
        self.start_time = time.time()
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initial progress update
        if progress_callback:
            progress_callback(0.02, "Initializing processing pipeline...")
        
        try:
            # Stage 1: Analyze document to get metadata
            if progress_callback:
                progress_callback(0.05, "Analyzing document structure...")
            
            document_info = self._analyze_document(document_text)
            
            # Stage 2: Generate processing plan using Planner agent
            if progress_callback:
                progress_callback(0.1, "Creating analysis plan...")
            
            # Get processing options
            process_options = self._get_processing_options(options)
            
            # Extract user preferences
            user_preferences = {
                "detail_level": process_options.detail_level,
                "focus_areas": process_options.focus_areas,
                "user_instructions": process_options.user_instructions
            }
            
            # Use first crew type
            crew_type = process_options.crews[0] if process_options.crews else "issues"
            
            # Create plan
            plan = self._create_plan(document_info, user_preferences, crew_type)
            
            if progress_callback:
                progress_callback(0.15, "Analysis plan created")
            
            # Stage 3: Process with appropriate crew
            if crew_type == "issues":
                if progress_callback:
                    progress_callback(0.2, "Starting issues analysis...")
                
                # Import here to avoid circular imports
                from crews.issues_crew import IssuesCrew
                
                # Create issues crew if not cached
                if "issues" not in self._crews:
                    self._crews["issues"] = IssuesCrew(
                        llm_client=self.llm_client,
                        verbose=self.verbose,
                        max_chunk_size=self.max_chunk_size,
                        max_rpm=self.max_rpm
                    )
                
                # Get the crew
                issues_crew = self._crews["issues"]
                
                # Process with issues crew
                crew_result = issues_crew.process_document(
                    document_text, 
                    document_info=document_info, 
                    user_preferences={
                        **user_preferences,
                        "agent_instructions": plan
                    },
                    max_chunk_size=process_options.max_chunk_size,
                    min_chunks=process_options.min_chunks,
                    enable_reviewer=process_options.enable_reviewer,
                    progress_callback=progress_callback
                )
                
                results = {"issues": crew_result}
            else:
                # Other crew types would be implemented here
                results = {"error": f"Crew type '{crew_type}' not implemented"}
            
            # Final completion
            if progress_callback:
                progress_callback(1.0, "Processing complete")
            
            # Add overall metadata
            processing_time = time.time() - self.start_time
            results["_metadata"] = {
                "run_id": self.run_id,
                "processing_time": processing_time,
                "document_info": document_info,
                "plan": plan,
                "options": process_options.to_dict() if hasattr(process_options, "to_dict") else process_options
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in document processing: {e}")
            if progress_callback:
                progress_callback(1.0, f"Error: {str(e)}")
            
            return {
                "error": str(e),
                "_metadata": {
                    "run_id": self.run_id,
                    "error": True,
                    "processing_time": time.time() - self.start_time
                }
            }
    
    def _analyze_document(self, document_text: str) -> Dict[str, Any]:
        """
        Analyze document to extract metadata.
        
        Args:
            document_text: Document text
            
        Returns:
            Document info dictionary
        """
        document_info = asyncio.run(self.analyzer.analyze_preview(document_text))
        document_info["original_text_length"] = len(document_text)
        return document_info
    
    def _create_plan(
        self, 
        document_info: Dict[str, Any], 
        user_preferences: Dict[str, Any],
        crew_type: str
    ) -> Dict[str, Any]:
        """
        Create processing plan using the Planner agent.
        
        Args:
            document_info: Document metadata
            user_preferences: User preferences
            crew_type: Type of crew
            
        Returns:
            Plan dictionary
        """
        return self.planner.create_plan(
            document_info=document_info,
            user_preferences=user_preferences,
            crew_type=crew_type
        )
    
    def _get_processing_options(self, options) -> ProcessingOptions:
        """
        Get processing options from provided options or defaults.
        
        Args:
            options: Options dictionary or ProcessingOptions object
            
        Returns:
            ProcessingOptions object
        """
        if options is None:
            # Use defaults
            return self.config_manager.get_processing_options()
        elif isinstance(options, dict):
            # Convert dictionary to ProcessingOptions
            return self.config_manager.create_options_from_dict(options)
        else:
            # Assume it's already a ProcessingOptions object
            return options