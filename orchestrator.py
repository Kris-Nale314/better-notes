"""
Integrated orchestrator for document processing and multi-agent analysis.
Manages both traditional summarization and agent-based analysis workflows.
"""

from typing import Dict, Any, List, Optional, Callable, Union
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import lean components
from lean.async_openai_adapter import AsyncOpenAIAdapter
from lean.document import DocumentAnalyzer
from lean.chunker import DocumentChunker
from lean.summarizer import ChunkSummarizer
from lean.synthesizer import Synthesizer
from lean.booster import Booster
from lean.options import ProcessingOptions

# Import crew components
from crews.issues_crew import IssuesCrew
# Future imports:
# from crews.action_crew import ActionCrew
# from crews.insights_crew import InsightsCrew

logger = logging.getLogger(__name__)

class OrchestratorFactory:
    """Factory for creating orchestrator instances with different capabilities."""
    
    @staticmethod
    def create_orchestrator(
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.2,
        max_chunk_size: int = 1500,
        verbose: bool = True,
        max_concurrent_chunks: int = 5,
        max_rpm: int = 10
    ) -> "Orchestrator":
        """
        Create an orchestrator with specified settings.
        
        Args:
            api_key: OpenAI API key (optional if set in environment)
            model: Model name to use
            temperature: Temperature parameter
            max_chunk_size: Maximum chunk size
            verbose: Whether to enable verbose output
            max_concurrent_chunks: Maximum concurrent chunks for processing
            max_rpm: Maximum requests per minute
            
        Returns:
            Configured Orchestrator instance
        """
        # Initialize the LLM client
        llm_client = AsyncOpenAIAdapter(
            model=model,
            api_key=api_key,
            temperature=temperature
        )
        
        # Create core components
        analyzer = DocumentAnalyzer(llm_client)
        chunker = DocumentChunker()
        
        # Create booster if needed for parallel processing
        if max_concurrent_chunks > 1:
            booster = Booster(
                cache_dir=".cache",
                max_workers=max_concurrent_chunks,
                enable_caching=False
            )
        else:
            booster = None
        
        # Create the orchestrator
        orchestrator = Orchestrator(
            llm_client=llm_client,
            analyzer=analyzer,
            chunker=chunker,
            verbose=verbose,
            max_chunk_size=max_chunk_size,
            max_rpm=max_rpm,
            booster=booster
        )
        
        return orchestrator
    
    @staticmethod
    def create_from_options(options: ProcessingOptions, api_key: Optional[str] = None) -> "Orchestrator":
        """
        Create an orchestrator from ProcessingOptions.
        
        Args:
            options: Processing options
            api_key: OpenAI API key (optional if set in environment)
            
        Returns:
            Configured Orchestrator instance
        """
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
    Unified orchestrator for document processing.
    Handles both traditional summarization and multi-agent analysis.
    """
    
    def __init__(
        self,
        llm_client: AsyncOpenAIAdapter,
        analyzer: DocumentAnalyzer,
        chunker: DocumentChunker,
        verbose: bool = True,
        max_chunk_size: int = 1500,
        max_rpm: int = 10,
        booster: Optional[Booster] = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            llm_client: LLM client for agent communication
            analyzer: Document analyzer
            chunker: Document chunker
            verbose: Whether to enable verbose output
            max_chunk_size: Maximum chunk size
            max_rpm: Maximum requests per minute
            booster: Optional booster for parallel processing
        """
        self.llm_client = llm_client
        self.analyzer = analyzer
        self.chunker = chunker
        self.verbose = verbose
        self.max_chunk_size = max_chunk_size
        self.max_rpm = max_rpm
        self.booster = booster
        
        # On-demand components - created when needed
        self._summarizer = None
        self._synthesizer = None
        self._issues_crew = None
        
        # Initialize refiner if available
        try:
            from ui_utils.refiner import SummaryRefiner
            self.refiner = SummaryRefiner(llm_client)
            self.has_refiner = True
        except ImportError:
            self.has_refiner = False
            logger.info("SummaryRefiner not available, refinement features disabled")
    
    @property
    def summarizer(self) -> ChunkSummarizer:
        """Get or create the chunk summarizer."""
        if self._summarizer is None:
            self._summarizer = ChunkSummarizer(self.llm_client)
        return self._summarizer
    
    @property
    def synthesizer(self) -> Synthesizer:
        """Get or create the synthesizer."""
        if self._synthesizer is None:
            self._synthesizer = Synthesizer(self.llm_client)
        return self._synthesizer
    
    @property
    def issues_crew(self) -> IssuesCrew:
        """Get or create the issues crew."""
        if self._issues_crew is None:
            self._issues_crew = IssuesCrew(
                llm_client=self.llm_client,
                verbose=self.verbose,
                max_chunk_size=self.max_chunk_size
            )
        return self._issues_crew
    
    def process_document(
        self,
        document_text: str,
        options: Dict[str, Any] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process a document with specified options.
        
        Args:
            document_text: Text to process
            options: Processing options (crews to use, chunk size, etc.)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary of results
        """
        options = options or {}
        
        # Choose processing path based on options
        if "crews" in options and options["crews"]:
            # Multi-agent processing path
            return self._process_with_crews(document_text, options, progress_callback)
        else:
            # Traditional summarization path
            return self._process_with_summarization(document_text, options, progress_callback)
    
    def _process_with_crews(
        self,
        document_text: str,
        options: Dict[str, Any],
        progress_callback: Optional[Callable]
    ) -> Dict[str, Any]:
        """
        Process a document using agent crews.
        
        Args:
            document_text: Text to process
            options: Processing options
            progress_callback: Progress callback
            
        Returns:
            Dictionary of results by crew type
        """
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
        
        # Create basic document info - limit preview size to avoid large headers
        preview_length = min(2000, len(document_text))
        document_preview = document_text[:preview_length]
        
        # Create document info
        document_info = {
            "original_text_length": len(document_text),
            "preview_text": document_preview
        }
        
        # Chunk document
        if self.verbose:
            logger.info(f"Chunking document with min_chunks={min_chunks}, max_chunk_size={max_chunk_size}")
        
        chunks = self.chunker.chunk_document(
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
                    # Update RPM setting if needed
                    if "max_rpm" in options and hasattr(self.issues_crew, "update_rpm"):
                        self.issues_crew.update_rpm(options.get("max_rpm", self.max_rpm))
                        
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
    
    def _process_with_summarization(
        self,
        document_text: str,
        options: Dict[str, Any],
        progress_callback: Optional[Callable]
    ) -> Dict[str, Any]:
        """
        Process a document using traditional summarization.
        
        Args:
            document_text: Text to process
            options: Processing options
            progress_callback: Progress callback
            
        Returns:
            Dictionary with summarization results
        """
        # Convert options to ProcessingOptions if needed
        if not isinstance(options, ProcessingOptions):
            processing_options = ProcessingOptions(
                model_name=self.llm_client.model,
                temperature=self.llm_client.temperature,
                min_chunks=options.get("min_chunks", 3),
                max_chunk_size=options.get("max_chunk_size", self.max_chunk_size),
                detail_level=options.get("detail_level", "detailed"),
                max_concurrent_chunks=options.get("max_concurrent_chunks", 5),
                user_instructions=options.get("user_preferences", {}).get("user_instructions")
            )
        else:
            processing_options = options
        
        # Use the traditional summarization path
        return self.process_document_sync(document_text, processing_options, progress_callback)
    
    async def process_document_async(
        self,
        document_text: str,
        options: Union[Dict[str, Any], ProcessingOptions] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process a document asynchronously.
        
        Args:
            document_text: Text to process
            options: Processing options or dictionary
            progress_callback: Progress callback
            
        Returns:
            Processing results
        """
        # Use ThreadPoolExecutor to run the synchronous method
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor,
                lambda: self.process_document(document_text, options, progress_callback)
            )
    
    def process_document_sync(
        self,
        document_text: str,
        options: Optional[ProcessingOptions] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process a document with the traditional summarization pipeline.
        This is a compatibility method for the lean.orchestrator interface.
        
        Args:
            document_text: The document text
            options: Processing options
            progress_callback: Progress callback
            
        Returns:
            Summarization results
        """
        from lean.orchestrator import Orchestrator as LeanOrchestrator
        
        # Create a temporary lean orchestrator
        temp_orchestrator = LeanOrchestrator(
            llm_client=self.llm_client,
            analyzer=self.analyzer,
            chunker=self.chunker,
            summarizer=self.summarizer,
            synthesizer=self.synthesizer,
            booster=self.booster,
            pass_processors={},
            options=options or ProcessingOptions()
        )
        
        # Process using the lean orchestrator
        return temp_orchestrator.process_document_sync(document_text, progress_callback)
    
    def update_max_chunk_size(self, new_size: int) -> None:
        """
        Update the maximum chunk size for all components.
        
        Args:
            new_size: New maximum chunk size
        """
        self.max_chunk_size = new_size
        
        # Update crews if they exist
        if self._issues_crew:
            self._issues_crew.max_chunk_size = new_size
    
    def update_max_rpm(self, new_rpm: int) -> None:
        """
        Update the maximum requests per minute for all components.
        
        Args:
            new_rpm: New maximum requests per minute
        """
        self.max_rpm = new_rpm
        
        # Update agents in issues crew if it exists
        if self._issues_crew:
            for agent_type in ["extractor", "aggregator", "evaluator", "formatter"]:
                if hasattr(self._issues_crew, f"{agent_type}_agent"):
                    agent = getattr(self._issues_crew, f"{agent_type}_agent")
                    if hasattr(agent, "agent") and hasattr(agent.agent, "max_rpm"):
                        agent.agent.max_rpm = new_rpm