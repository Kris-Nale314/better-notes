"""
Orchestrator - Coordinates the document processing pipeline using the new architecture.
Manages the flow of data through the multi-agent system with improved error handling.
"""

from typing import Dict, Any, List, Optional, Union, Callable
import asyncio
import logging
import traceback
import time
import importlib
from datetime import datetime

# Import our universal adapter and config manager
from universal_llm_adapter import UniversalLLMAdapter
from config_manager import ConfigManager, ProcessingOptions

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Unified orchestrator for document processing.
    Coordinates document analysis, chunking, and multi-agent processing.
    """
    
    def __init__(self,
                llm_client: UniversalLLMAdapter,
                analyzer=None,
                chunker=None,
                planner=None,
                verbose: bool = True,
                max_chunk_size: int = 10000,
                max_rpm: int = 10,
                config_manager: Optional[ConfigManager] = None):
        """
        Initialize the orchestrator with all necessary components.
        
        Args:
            llm_client: Universal LLM adapter
            analyzer: Document analyzer (loaded dynamically if None)
            chunker: Document chunker (loaded dynamically if None)
            planner: Planner agent (loaded dynamically if None)
            verbose: Whether to enable verbose logging
            max_chunk_size: Maximum chunk size
            max_rpm: Maximum requests per minute
            config_manager: Config manager (created if None)
        """
        self.llm_client = llm_client
        self.verbose = verbose
        self.max_chunk_size = max_chunk_size
        self.max_rpm = max_rpm
        
        # Create or use provided config manager
        self.config_manager = config_manager or ConfigManager()
        
        # Load or use provided components
        self.analyzer = analyzer or self._load_analyzer()
        self.chunker = chunker or self._load_chunker()
        self.planner = planner or self._load_planner()
        
        # For tracking current process
        self.run_id = None
        self.start_time = None
        
        # On-demand components
        self._issues_crew = None
        
        logger.info("Orchestrator initialized successfully")
    
    def _load_analyzer(self):
        """Dynamically load document analyzer."""
        try:
            # Try different import paths
            try:
                from lean.document import DocumentAnalyzer
            except ImportError:
                try:
                    from document import DocumentAnalyzer
                except ImportError:
                    # Try relative import as last resort
                    module = importlib.import_module('.document', package='lean')
                    DocumentAnalyzer = getattr(module, 'DocumentAnalyzer')
            
            return DocumentAnalyzer(self.llm_client)
        except Exception as e:
            logger.error(f"Error loading document analyzer: {e}")
            raise ImportError(f"Could not load DocumentAnalyzer: {e}")
    
    def _load_chunker(self):
        """Dynamically load document chunker."""
        try:
            # Try different import paths
            try:
                from lean.chunker import DocumentChunker
            except ImportError:
                try:
                    from chunker import DocumentChunker
                except ImportError:
                    # Try relative import as last resort
                    module = importlib.import_module('.chunker', package='lean')
                    DocumentChunker = getattr(module, 'DocumentChunker')
            
            return DocumentChunker()
        except Exception as e:
            logger.error(f"Error loading document chunker: {e}")
            raise ImportError(f"Could not load DocumentChunker: {e}")
    
    def _load_planner(self):
        """Dynamically load planner agent."""
        try:
            # Try different import paths
            try:
                from agents.planner import PlannerAgent
            except ImportError:
                try:
                    from planner import PlannerAgent
                except ImportError:
                    # Try relative import as last resort
                    module = importlib.import_module('.planner', package='agents')
                    PlannerAgent = getattr(module, 'PlannerAgent')
            
            # Load planner config
            planner_config = self.config_manager.get_config("planner")
            
            return PlannerAgent(
                llm_client=self.llm_client,
                config=planner_config,
                verbose=self.verbose,
                max_chunk_size=self.max_chunk_size,
                max_rpm=self.max_rpm
            )
        except Exception as e:
            logger.error(f"Error loading planner agent: {e}")
            raise ImportError(f"Could not load PlannerAgent: {e}")
    
    @property
    def issues_crew(self):
        """Lazy-load issues crew."""
        if self._issues_crew is None:
            try:
                # Try different import paths
                try:
                    from crews.issues_crew import IssuesCrew
                except ImportError:
                    try:
                        from issues_crew import IssuesCrew
                    except ImportError:
                        # Try relative import as last resort
                        module = importlib.import_module('.issues_crew', package='crews')
                        IssuesCrew = getattr(module, 'IssuesCrew')
                
                # Get config
                issues_config = self.config_manager.get_config("issues")
                
                # Try to determine the correct parameters for IssuesCrew
                try:
                    # First try with the new interface we designed
                    self._issues_crew = IssuesCrew(
                        llm_client=self.llm_client,
                        config=issues_config,
                        config_manager=self.config_manager,
                        verbose=self.verbose,
                        max_chunk_size=self.max_chunk_size,
                        max_rpm=self.max_rpm
                    )
                except TypeError as e:
                    # If that fails, try the original interface without config and config_manager
                    if "got an unexpected keyword argument 'config'" in str(e):
                        logger.info("Using original IssuesCrew interface without config parameter")
                        self._issues_crew = IssuesCrew(
                            llm_client=self.llm_client,
                            verbose=self.verbose,
                            max_chunk_size=self.max_chunk_size,
                            max_rpm=self.max_rpm
                        )
                    elif "got an unexpected keyword argument 'config_manager'" in str(e):
                        logger.info("Using original IssuesCrew interface without config_manager parameter")
                        self._issues_crew = IssuesCrew(
                            llm_client=self.llm_client,
                            config=issues_config,
                            verbose=self.verbose,
                            max_chunk_size=self.max_chunk_size,
                            max_rpm=self.max_rpm
                        )
                    else:
                        # Re-raise if it's another error
                        raise
            except Exception as e:
                logger.error(f"Error loading issues crew: {e}")
                raise ImportError(f"Could not load IssuesCrew: {e}")
        
        return self._issues_crew
    
    def process_document(self,
                         document_text: str,
                         options: Optional[Union[Dict[str, Any], ProcessingOptions]] = None,
                         progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process a document through the pipeline.
        
        Args:
            document_text: Document text to process
            options: Processing options (dictionary or ProcessingOptions)
            progress_callback: Callback for progress updates
            
        Returns:
            Processing results
        """
        # Convert options dictionary to ProcessingOptions if needed
        if options is None:
            process_options = self.config_manager.get_processing_options()
        elif isinstance(options, dict):
            # Extract user preferences from options dict for backward compatibility
            user_preferences = options.get("user_preferences", {})
            
            # Create ProcessingOptions with values from options dict
            process_options = ProcessingOptions(
                model_name=options.get("model_name", "gpt-3.5-turbo"),
                temperature=options.get("temperature", 0.2),
                min_chunks=options.get("min_chunks", 3),
                max_chunk_size=options.get("max_chunk_size", self.max_chunk_size),
                detail_level=user_preferences.get("detail_level", "standard"),
                user_instructions=user_preferences.get("user_instructions", ""),
                focus_areas=user_preferences.get("focus_areas", []),
                crews=options.get("crews", ["issues"]),
                enable_reviewer=options.get("enable_reviewer", True),
                max_rpm=options.get("max_rpm", self.max_rpm)
            )
        else:
            process_options = options
        
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
                
            user_preferences = {
                "detail_level": process_options.detail_level,
                "focus_areas": process_options.focus_areas,
                "user_instructions": process_options.user_instructions
            }
            
            plan = self._create_plan(document_info, user_preferences, process_options.crews[0])
            
            if progress_callback:
                progress_callback(0.15, "Analysis plan created")
            
            # Stage 3: Process with appropriate crews
            results = {}
            progress_per_crew = 0.8 / len(process_options.crews)
            current_progress = 0.2
            
            for crew_type in process_options.crews:
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
                        process_options, 
                        crew_progress_wrapper
                    )
                    
                    results[crew_type] = crew_result
                    current_progress = crew_end
                
                # Add more crew types here as they are implemented
            
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
            logger.error(f"Error in document processing: {str(e)}")
            if progress_callback:
                progress_callback(1.0, f"Error: {str(e)}")
            
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
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
        options: ProcessingOptions,
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
        user_preferences = {
            "detail_level": options.detail_level,
            "focus_areas": options.focus_areas,
            "user_instructions": options.user_instructions,
        }
        
        # Check if the issues_crew can handle agent_instructions - try to add it if possible
        try:
            # Create a test dictionary with agent_instructions
            test_prefs = user_preferences.copy()
            test_prefs["agent_instructions"] = plan
            
            # Try to pass to a method that handles user_preferences
            if hasattr(self.issues_crew, "process_document"):
                # Just check the signature, don't actually call it
                self.issues_crew.process_document.__code__
                
                # If we get here, add the agent_instructions to the real preferences
                user_preferences["agent_instructions"] = plan
                logger.info("Added agent_instructions to user_preferences")
        except Exception as e:
            # If there's any error, just keep the original preferences
            logger.info(f"Keeping original user_preferences without agent_instructions: {str(e)}")
            pass
        
        # Update crew settings
        if hasattr(self.issues_crew, "update_rpm"):
            self.issues_crew.update_rpm(options.max_rpm)
        
        # Process with the issues crew
        result = self.issues_crew.process_document(
            document_text,
            document_info=document_info,
            user_preferences=user_preferences,
            max_chunk_size=options.max_chunk_size,
            min_chunks=options.min_chunks,
            enable_reviewer=options.enable_reviewer,
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
    
    def process_document_sync(self,
                            document_text: str,
                            options: Optional[Union[Dict[str, Any], ProcessingOptions]] = None,
                            progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process a document synchronously (convenience wrapper).
        
        Args:
            document_text: Document text to process
            options: Processing options
            progress_callback: Progress callback
            
        Returns:
            Processing results
        """
        return self.process_document(document_text, options, progress_callback)