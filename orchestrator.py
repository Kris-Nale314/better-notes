"""
Enhanced orchestrator for document processing with Planner-driven workflows.
"""

from typing import Dict, Any, List, Optional, Callable, Union
import logging
import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor

# Import lean components
from lean.async_openai_adapter import AsyncOpenAIAdapter
from lean.document import DocumentAnalyzer
from lean.chunker import DocumentChunker
from lean.summarizer import ChunkSummarizer
from lean.synthesizer import Synthesizer
from lean.booster import Booster
from lean.options import ProcessingOptions

# Import agent components
from agents.planner import PlannerAgent

# Import crew components
from crews.issues_crew import IssuesCrew

logger = logging.getLogger(__name__)

class OrchestratorFactory:
   @staticmethod
   def create_orchestrator(
       api_key: Optional[str] = None,
       model: str = "gpt-3.5-turbo",
       temperature: float = 0.2,
       max_chunk_size: int = 10000,  # Increased for macro-chunking
       verbose: bool = True,
       max_concurrent_chunks: int = 5,
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
       
       # Parallel processing booster
       booster = None
       if max_concurrent_chunks > 1:
           booster = Booster(
               cache_dir=".cache",
               max_workers=max_concurrent_chunks,
               enable_caching=False
           )
       
       return Orchestrator(
           llm_client=llm_client,
           analyzer=analyzer,
           chunker=chunker,
           planner=planner,
           verbose=verbose,
           max_chunk_size=max_chunk_size,
           max_rpm=max_rpm,
           booster=booster
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
   """Unified orchestrator with Planner-driven workflows."""
   
   def __init__(
       self,
       llm_client: AsyncOpenAIAdapter,
       analyzer: DocumentAnalyzer,
       chunker: DocumentChunker,
       planner: PlannerAgent,
       verbose: bool = True,
       max_chunk_size: int = 10000,
       max_rpm: int = 10,
       booster: Optional[Booster] = None
   ):
       """Initialize orchestrator with all components."""
       self.llm_client = llm_client
       self.analyzer = analyzer
       self.chunker = chunker
       self.planner = planner
       self.verbose = verbose
       self.max_chunk_size = max_chunk_size
       self.max_rpm = max_rpm
       self.booster = booster
       
       # On-demand components
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
           logger.info("SummaryRefiner not available")
   
   @property
   def summarizer(self) -> ChunkSummarizer:
       """Get or create summarizer."""
       if self._summarizer is None:
           self._summarizer = ChunkSummarizer(self.llm_client)
       return self._summarizer
   
   @property
   def synthesizer(self) -> Synthesizer:
       """Get or create synthesizer."""
       if self._synthesizer is None:
           self._synthesizer = Synthesizer(self.llm_client)
       return self._synthesizer
   
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
       """Process document with specified options."""
       options = options or {}
       
       # Choose processing path
       if "crews" in options and options["crews"]:
           return self._process_with_crews(document_text, options, progress_callback)
       else:
           return self._process_with_summarization(document_text, options, progress_callback)
   
   def _process_with_crews(
       self,
       document_text: str,
       options: Dict[str, Any],
       progress_callback: Optional[Callable]
   ) -> Dict[str, Any]:
       """Process with agent crews (Planner-driven)."""
       results = {}
       errors = []
       
       # Processing parameters
       crew_types = options.get("crews", ["issues"])
       min_chunks = options.get("min_chunks", 3)
       max_chunk_size = options.get("max_chunk_size", self.max_chunk_size)
       enable_reviewer = options.get("enable_reviewer", True)
       
       # User preferences
       user_preferences = options.get("user_preferences", {})
       user_preferences["min_chunks"] = min_chunks
       user_preferences["max_chunk_size"] = max_chunk_size
       
       # 1. Document Analysis
       try:
           if progress_callback:
               progress_callback(0.05, "Analyzing document...")
           
           document_info = asyncio.run(self.analyzer.analyze_preview(document_text))
           document_info["original_text_length"] = len(document_text)
           
           if progress_callback:
               progress_callback(0.1, "Document analysis complete")
       except Exception as e:
           logger.error(f"Document analysis error: {str(e)}")
           document_info = {"original_text_length": len(document_text)}
           errors.append(f"Document analysis: {str(e)}")
       
       # 2. Generate plans for requested crews
       try:
           if progress_callback:
               progress_callback(0.15, "Creating document analysis plan...")
           
           plan = self.planner.create_plan(
               document_info=document_info,
               user_preferences=user_preferences,
               crew_type=crew_types[0]  # Focus on primary crew
           )
           
           if progress_callback:
               progress_callback(0.2, "Planning complete")
       except Exception as e:
           logger.error(f"Planning error: {str(e)}")
           plan = {}
           errors.append(f"Planning: {str(e)}")
       
       # 3. Process with each crew
       current_progress = 0.2
       progress_per_crew = 0.8 / len(crew_types)
       
       for crew_type in crew_types:
           if progress_callback:
               progress_callback(current_progress, f"Starting {crew_type} analysis...")
           
           try:
               # Process with appropriate crew
               if crew_type == "issues":
                   # Update crew settings
                   if "max_rpm" in options:
                       self.issues_crew.update_rpm(options.get("max_rpm", self.max_rpm))
                   
                   # Progress wrapper
                   def crew_progress_callback(progress, message):
                       if progress_callback:
                           scaled_progress = current_progress + (progress * progress_per_crew)
                           progress_callback(scaled_progress, message)
                   
                   # Process with issues crew
                   issues_result = self.issues_crew.process_document(
                       document_text,
                       document_info=document_info,
                       user_preferences=user_preferences,
                       max_chunk_size=max_chunk_size,
                       enable_reviewer=enable_reviewer,
                       progress_callback=crew_progress_callback
                   )
                   
                   results[crew_type] = issues_result
               else:
                   # Placeholder for future crews
                   results[crew_type] = f"{crew_type} analysis not implemented"
               
               if progress_callback:
                   current_progress += progress_per_crew
                   progress_callback(current_progress, f"{crew_type} analysis complete")
           
           except Exception as e:
               logger.error(f"{crew_type} analysis error: {str(e)}", exc_info=True)
               results[crew_type] = {"error": str(e), "traceback": traceback.format_exc()}
               errors.append(f"{crew_type} analysis: {str(e)}")
               
               if progress_callback:
                   current_progress += progress_per_crew
                   progress_callback(current_progress, f"{crew_type} analysis failed: {str(e)}")
       
       # Add errors to results
       if errors:
           results["_errors"] = errors
       
       # Add metadata
       results["_metadata"] = {
           "document_length": len(document_text),
           "crew_types": crew_types,
           "plan_generated": bool(plan)
       }
       
       if progress_callback:
           progress_callback(1.0, "Processing complete")
       
       return results
   
   def _process_with_summarization(
       self,
       document_text: str,
       options: Dict[str, Any],
       progress_callback: Optional[Callable]
   ) -> Dict[str, Any]:
       """Process with traditional summarization pipeline."""
       # Convert to ProcessingOptions
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
       
       return self.process_document_sync(document_text, processing_options, progress_callback)
   
   async def process_document_async(
       self,
       document_text: str,
       options: Union[Dict[str, Any], ProcessingOptions] = None,
       progress_callback: Optional[Callable] = None
   ) -> Dict[str, Any]:
       """Process document asynchronously."""
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
       """Process with traditional summarization."""
       from lean.orchestrator import Orchestrator as LeanOrchestrator
       
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
       
       return temp_orchestrator.process_document_sync(document_text, progress_callback)
   
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