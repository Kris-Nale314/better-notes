# orchestrator.py
from typing import Dict, Any, List, Optional
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from crews.issues_crew import IssuesCrew
# When implemented:
# from crews.action_crew import ActionCrew
# from crews.opp_crew import OppCrew

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Coordinates multiple agent crews to analyze documents in parallel.
    """
    
    def __init__(self, llm_client, verbose=True):
        """
        Initialize the orchestrator.
        
        Args:
            llm_client: LLM client for agent communication
            verbose: Whether to enable verbose output
        """
        self.llm_client = llm_client
        self.verbose = verbose
        
        # Initialize crews
        self.issues_crew = IssuesCrew(llm_client, verbose=verbose)
        # When implemented:
        # self.action_crew = ActionCrew(llm_client, verbose=verbose)
        # self.opp_crew = OppCrew(llm_client, verbose=verbose)
    
    def process_document(self, document_text, options=None, progress_callback=None):
        """
        Process a document with one or more crews.
        
        Args:
            document_text: Text to process
            options: Processing options (crews to use, etc.)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary of results by crew type
        """
        from lean.chunker import DocumentChunker
        from lean.document import DocumentAnalyzer
        
        options = options or {}
        
        # Initialize results
        results = {}
        
        # Analyze document to get metadata
        if self.verbose:
            logger.info("Analyzing document...")
        
        # Initialize document analyzer
        doc_analyzer = DocumentAnalyzer(self.llm_client)
        
        # Get document info
        document_info = {
            "original_text_length": len(document_text),
            # We'd normally await this, but simplifying for this example
            # preview_analysis: await doc_analyzer.analyze_preview(document_text)
        }
        
        # Chunk document
        if self.verbose:
            logger.info("Chunking document...")
        
        chunker = DocumentChunker()
        chunks = chunker.chunk_document(
            document_text,
            min_chunks=options.get("min_chunks", 3),
            max_chunk_size=options.get("max_chunk_size", None)
        )
        
        # Extract chunk texts
        document_chunks = [chunk["text"] for chunk in chunks]
        
        # Run requested crews
        crew_types = options.get("crews", ["issues"])
        
        if "issues" in crew_types:
            if progress_callback:
                progress_callback(0.1, "Identifying issues...")
            
            if self.verbose:
                logger.info("Running issues crew...")
            
            issues_result = self.issues_crew.process_document(
                document_chunks, 
                document_info,
                options.get("user_preferences", {})
            )
            
            results["issues"] = issues_result
            
            if progress_callback:
                progress_callback(0.4, "Issues analysis complete")
        
        # When implemented:
        # if "actions" in crew_types:
        #     if progress_callback:
        #         progress_callback(0.5, "Extracting action items...")
        #     
        #     if self.verbose:
        #         logger.info("Running action items crew...")
        #     
        #     action_result = self.action_crew.process_document(
        #         document_chunks, 
        #         document_info,
        #         options.get("user_preferences", {})
        #     )
        #     
        #     results["actions"] = action_result
        #     
        #     if progress_callback:
        #         progress_callback(0.7, "Action items extraction complete")
        # 
        # if "opportunities" in crew_types:
        #     if progress_callback:
        #         progress_callback(0.8, "Discovering opportunities...")
        #     
        #     if self.verbose:
        #         logger.info("Running opportunities crew...")
        #     
        #     opp_result = self.opp_crew.process_document(
        #         document_chunks, 
        #         document_info,
        #         options.get("user_preferences", {})
        #     )
        #     
        #     results["opportunities"] = opp_result
        #     
        #     if progress_callback:
        #         progress_callback(0.9, "Opportunities discovery complete")
        
        if progress_callback:
            progress_callback(1.0, "Processing complete")
        
        return results
    
    async def process_document_async(self, document_text, options=None, progress_callback=None):
        """
        Process a document asynchronously with one or more crews.
        
        Args:
            document_text: Text to process
            options: Processing options (crews to use, etc.)
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
    
    def run_multiple_crews(self, document_text, crew_types=None, options=None):
        """
        Run multiple crews in parallel on the same document.
        
        Args:
            document_text: Text to process
            crew_types: List of crew types to run (e.g., ["issues", "actions"])
            options: Processing options
            
        Returns:
            Dictionary of results by crew type
        """
        crew_types = crew_types or ["issues"]
        options = options or {}
        
        # Set crews in options
        options["crews"] = crew_types
        
        # Process document with all requested crews
        return self.process_document(document_text, options)