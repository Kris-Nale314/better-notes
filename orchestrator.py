"""
Enhanced orchestrator for document processing with Planner-driven workflows.
Coordinates the agent crews and manages the processing pipeline.
"""

from typing import Dict, Any, List, Optional, Callable, Union
import logging
import asyncio
import traceback
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Import lean components
from lean.async_openai_adapter import AsyncOpenAIAdapter
from lean.document import DocumentAnalyzer
from lean.chunker import DocumentChunker
from lean.options import ProcessingOptions

# Import agent components
from agents.planner import PlannerAgent

# Import crew components
from crews.issues_crew import IssuesCrew

logger = logging.getLogger(__name__)

class OrchestratorFactory:
    """Factory for creating properly configured orchestrator instances."""
    
    @staticmethod
    def create_orchestrator(
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.2,
        max_chunk_size: int = 10000,
        verbose: bool = True,
        max_rpm: int = 10
    ) -> "Orchestrator":
        """Create a configured orchestrator instance."""
        # Initialize the LLM client
        llm_client = AsyncOpenAIAdapter(
            model=model,
            api_key=api_key,
            temperature=temperature
        )
        
        # Core components
        analyzer = DocumentAnalyzer(llm_client)
        chunker = DocumentChunker()
        planner = PlannerAgent(llm_client, verbose=verbose, max_rpm=max_rpm)
        
        return Orchestrator(
            llm_client=llm_client,
            analyzer=analyzer,
            chunker=chunker,
            planner=planner,
            verbose=verbose,
            max_chunk_size=max_chunk_size,
            max_rpm=max_rpm
        )
    
    @staticmethod
    def create_from_options(options: ProcessingOptions, api_key: Optional[str] = None) -> "Orchestrator":
        """Create orchestrator from ProcessingOptions."""
        return OrchestratorFactory.create_orchestrator(
            api_key=api_key,
            model=options.model_name,
            temperature=options.temperature,
            max_chunk_size=options.max_chunk_size,
            max_concurrent_chunks=options.max_concurrent_chunks,
            verbose=True
        )

class Orchestrator:
    """
    Unified orchestrator with Planner-driven workflows.
    Manages the entire document processing pipeline.
    """
    
    def __init__(
        self,
        llm_client: AsyncOpenAIAdapter,
        analyzer: DocumentAnalyzer,
        chunker: DocumentChunker,
        planner: PlannerAgent,
        verbose: bool = True,
        max_chunk_size: int = 10000,
        max_rpm: int = 10
    ):
        """Initialize orchestrator with all components."""
        self.llm_client = llm_client
        self.analyzer = analyzer
        self.chunker = chunker
        self.planner = planner
        self.verbose = verbose
        self.max_chunk_size = max_chunk_size
        self.max_rpm = max_rpm
        
        # On-demand components
        self._issues_crew = None
        self.run_id = None
    
    @property
    def issues_crew(self) -> IssuesCrew:
        """Get or create issues crew."""
        if self._issues_crew is None:
            self._issues_crew = IssuesCrew(
                llm_client=self.llm_client,
                verbose=self.verbose,
                max_chunk_size=self.max_chunk_size,
                max_rpm=self.max_rpm
            )
        return self._issues_crew
    
    def process_document(
        self,
        document_text: str,
        options: Dict[str, Any] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process document with specified options.
        
        Args:
            document_text: The document text to process
            options: Processing options
            progress_callback: Callback for progress updates
            
        Returns:
            Processing results
        """
        # Generate a unique run ID
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = time.time()
        
        # Default options
        options = options or {}
        crew_types = options.get("crews", ["issues"])
        
        # Progress tracking
        if progress_callback:
            progress_callback(0.02, "Initializing processing pipeline...")
        
        try:
            # Analyze document to get metadata
            if progress_callback:
                progress_callback(0.05, "Analyzing document structure...")
            
            document_info = self._analyze_document(document_text)
            
            # Generate processing plan using Planner agent
            if progress_callback:
                progress_callback(0.1, "Creating analysis plan...")
                
            user_preferences = options.get("user_preferences", {})
            plan = self._create_plan(document_info, user_preferences, crew_types[0])
            
            if progress_callback:
                progress_callback(0.15, "Analysis plan created")
            
            # Process with appropriate crews
            results = {}
            progress_per_crew = 0.8 / len(crew_types)
            current_progress = 0.2
            
            for crew_type in crew_types:
                crew_start = current_progress
                crew_end = crew_start + progress_per_crew
                
                # Wrap progress callback to scale for this crew
                def crew_progress_wrapper(crew_progress, message):
                    if progress_callback:
                        # Scale progress to the crew's allocated range
                        overall_progress = crew_start + (crew_progress * progress_per_crew)
                        progress_callback(overall_progress, message)
                
                # Process with the appropriate crew
                if crew_type == "issues":
                    if progress_callback:
                        progress_callback(current_progress, "Starting issues analysis...")
                    
                    # Process with issues crew
                    crew_result = self._process_with_issues_crew(
                        document_text, 
                        document_info, 
                        plan,
                        options, 
                        crew_progress_wrapper
                    )
                    
                    results[crew_type] = crew_result
                    current_progress = crew_end
                
                # Add more crew types here as they are implemented
            
            # Final completion
            if progress_callback:
                progress_callback(1.0, "Processing complete")
            
            # Add overall metadata
            processing_time = time.time() - start_time
            results["_metadata"] = {
                "run_id": self.run_id,
                "processing_time": processing_time,
                "document_info": document_info,
                "plan": plan,
                "options": options
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in document processing: {str(e)}")
            if progress_callback:
                progress_callback(1.0, f"Error: {str(e)}")
            
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "_metadata": {
                    "run_id": self.run_id,
                    "error": True,
                    "processing_time": time.time() - start_time
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
        try:
            document_info = asyncio.run(self.analyzer.analyze_preview(document_text))
            document_info["original_text_length"] = len(document_text)
            return document_info
        except Exception as e:
            logger.warning(f"Document analysis failed: {str(e)}")
            # Return basic info if analysis fails
            return {
                "original_text_length": len(document_text),
                "basic_stats": {
                    "word_count": len(document_text.split()),
                    "char_count": len(document_text)
                }
            }
    
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
        try:
            return self.planner.create_plan(
                document_info=document_info,
                user_preferences=user_preferences,
                crew_type=crew_type
            )
        except Exception as e:
            logger.warning(f"Plan creation failed: {str(e)}")
            # Return empty plan if creation fails
            return {}
    
    def _process_with_issues_crew(
        self, 
        document_text: str,
        document_info: Dict[str, Any],
        plan: Dict[str, Any],
        options: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process document with the Issues Crew.
        
        Args:
            document_text: Document text
            document_info: Document metadata
            plan: Processing plan
            options: Processing options
            progress_callback: Progress callback
            
        Returns:
            Processing results
        """
        # Extract options
        user_preferences = options.get("user_preferences", {})
        max_chunk_size = options.get("max_chunk_size", self.max_chunk_size)
        min_chunks = options.get("min_chunks", 3)
        enable_reviewer = options.get("enable_reviewer", True)
        
        # Update crew settings
        self.issues_crew.update_rpm(options.get("max_rpm", self.max_rpm))
        
        # Process with the issues crew
        result = self.issues_crew.process_document(
            document_text,
            document_info=document_info,
            user_preferences=user_preferences,
            max_chunk_size=max_chunk_size,
            min_chunks=min_chunks,
            enable_reviewer=enable_reviewer,
            progress_callback=progress_callback
        )
        
        return result
    
    def update_max_chunk_size(self, new_size: int) -> None:
        """Update maximum chunk size for all components."""
        self.max_chunk_size = new_size
        
        if self._issues_crew:
            self._issues_crew.max_chunk_size = new_size
    
    def update_max_rpm(self, new_rpm: int) -> None:
        """Update maximum requests per minute for all components."""
        self.max_rpm = new_rpm
        
        if self._issues_crew:
            self._issues_crew.update_rpm(new_rpm)