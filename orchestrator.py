"""
Orchestrator for agent crews - coordinates document analysis across multiple agent teams.
"""

from typing import Dict, Any, List, Optional, Callable
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from crews.issues_crew import IssuesCrew
# Import when implemented:
# from crews.action_crew import ActionCrew
# from crews.insights_crew import InsightsCrew

from lean.chunker import DocumentChunker
from lean.document import DocumentAnalyzer

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Coordinates multiple agent crews to analyze documents in parallel.
    Manages document chunking, analysis distribution, and result aggregation.
    """
    
    def __init__(self, llm_client, verbose=True, max_chunk_size=1500):
        """
        Initialize the orchestrator.
        
        Args:
            llm_client: LLM client for agent communication
            verbose: Whether to enable verbose output
            max_chunk_size: Maximum size of text chunks to process
        """
        self.llm_client = llm_client
        self.verbose = verbose
        self.max_chunk_size = max_chunk_size
        
        # Initialize crews
        self.issues_crew = IssuesCrew(
            llm_client=llm_client, 
            verbose=verbose,
            max_chunk_size=max_chunk_size
        )
        
        # Initialize other crews when implemented
        # self.action_crew = ActionCrew(
        #     llm_client=llm_client, 
        #     verbose=verbose,
        #     max_chunk_size=max_chunk_size
        # )
        # self.insights_crew = InsightsCrew(
        #     llm_client=llm_client, 
        #     verbose=verbose,
        #     max_chunk_size=max_chunk_size
        # )
    
    def process_document(self, document_text: str, options: Dict[str, Any] = None, 
                        progress_callback: Callable = None) -> Dict[str, Any]:
        """
        Process a document with one or more crews.
        
        Args:
            document_text: Text to process
            options: Processing options (crews to use, chunk size, etc.)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary of results by crew type
        """
        options = options or {}
        
        # Initialize results
        results = {}
        
        # Extract processing parameters
        crew_types = options.get("crews", ["issues"])
        min_chunks = options.get("min_chunks", 3)
        max_chunk_size = options.get("max_chunk_size", self.max_chunk_size)
        user_preferences = options.get("user_preferences", {})
        
        # Update max chunk size if specified
        if max_chunk_size != self.max_chunk_size:
            self.max_chunk_size = max_chunk_size
        
        # Analyze document to get metadata
        if self.verbose:
            logger.info("Analyzing document...")
        
        # Initialize document analyzer
        doc_analyzer = DocumentAnalyzer(self.llm_client)
        
        # Create basic document info - limit preview size to avoid large headers
        preview_length = min(2000, len(document_text))
        document_preview = document_text[:preview_length]
        
        # Create document info
        document_info = {
            "original_text_length": len(document_text),
            "preview_text": document_preview
            # For async implementation, we would await:
            # "preview_analysis": await doc_analyzer.analyze_preview(document_text, preview_length)
        }
        
        # Chunk document
        if self.verbose:
            logger.info(f"Chunking document with min_chunks={min_chunks}, max_chunk_size={max_chunk_size}")
        
        chunker = DocumentChunker()
        chunks = chunker.chunk_document(
            document_text,
            min_chunks=min_chunks,
            max_chunk_size=max_chunk_size
        )
        
        # Extract chunk texts
        document_chunks = [chunk["text"] for chunk in chunks]
        
        # Update progress
        if progress_callback:
            progress_callback(0.1, f"Document prepared: {len(document_chunks)} chunks")
        
        # Run requested crews sequentially
        current_progress = 0.1
        progress_per_crew = 0.9 / len(crew_types)
        
        for crew_type in crew_types:
            # Start progress tracking for this crew
            if progress_callback:
                progress_callback(current_progress, f"Starting {crew_type} analysis...")
            
            # Run the appropriate crew
            if crew_type == "issues":
                if self.verbose:
                    logger.info(f"Running issues crew...")
                
                try:
                    issues_result = self.issues_crew.process_document(
                        document_chunks, 
                        document_info,
                        user_preferences,
                        max_chunk_size
                    )
                    
                    results["issues"] = issues_result
                    
                    if progress_callback:
                        current_progress += progress_per_crew
                        progress_callback(current_progress, "Issues analysis complete")
                        
                except Exception as e:
                    logger.error(f"Error in issues analysis: {str(e)}")
                    results["issues"] = {"error": str(e)}
                    if progress_callback:
                        progress_callback(current_progress, f"Issues analysis failed: {str(e)}")
            
            elif crew_type == "actions":
                # Will implement when ActionCrew is ready
                if self.verbose:
                    logger.info("Action items analysis not yet implemented")
                
                results["actions"] = "Action items analysis not yet implemented"
                
                if progress_callback:
                    current_progress += progress_per_crew
                    progress_callback(current_progress, "Action items placeholder (not implemented)")
            
            elif crew_type == "insights":
                # Will implement when InsightsCrew is ready
                if self.verbose:
                    logger.info("Context insights analysis not yet implemented")
                
                results["insights"] = "Context insights analysis not yet implemented"
                
                if progress_callback:
                    current_progress += progress_per_crew
                    progress_callback(current_progress, "Context insights placeholder (not implemented)")
        
        # Final progress update
        if progress_callback:
            progress_callback(1.0, "Processing complete")
        
        return results
    
    async def process_document_async(self, document_text: str, options: Dict[str, Any] = None, 
                                   progress_callback: Callable = None) -> Dict[str, Any]:
        """
        Process a document asynchronously with one or more crews.
        
        Args:
            document_text: Text to process
            options: Processing options (crews to use, chunk size, etc.)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary of results by crew type
        """
        # Use ThreadPoolExecutor to run the synchronous process_document method
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor,
                lambda: self.process_document(document_text, options, progress_callback)
            )
    
    def update_max_chunk_size(self, new_size: int) -> None:
        """
        Update the maximum chunk size for all crews.
        
        Args:
            new_size: New maximum chunk size
        """
        self.max_chunk_size = new_size
        
        # Update all crews
        self.issues_crew.max_chunk_size = new_size
        # When implemented:
        # self.action_crew.max_chunk_size = new_size
        # self.insights_crew.max_chunk_size = new_size
    
    def update_max_rpm(self, new_rpm: int) -> None:
        """
        Update the maximum requests per minute for all agents.
        
        Args:
            new_rpm: New maximum requests per minute
        """
        # Update for issues crew
        for agent_type in ["extractor", "aggregator", "evaluator", "formatter"]:
            if hasattr(self.issues_crew, f"{agent_type}_agent"):
                agent = getattr(self.issues_crew, f"{agent_type}_agent")
                if hasattr(agent, "agent") and hasattr(agent.agent, "max_rpm"):
                    agent.agent.max_rpm = new_rpm
        
        # When other crews are implemented, update them here