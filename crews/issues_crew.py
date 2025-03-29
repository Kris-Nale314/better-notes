"""
Enhanced Issues Crew with cleaner architecture.
Analyzes documents to identify issues, problems, risks, and challenges.
"""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

# Import agents
from agents.base import BaseAgent
from agents.planner import PlannerAgent
from agents.extractor import ExtractorAgent
from agents.aggregator import AggregatorAgent
from agents.evaluator import EvaluatorAgent
from agents.formatter import FormatterAgent
from agents.reviewer import ReviewerAgent

# Import document processing components
from lean.chunker import DocumentChunker
from lean.document import DocumentAnalyzer

logger = logging.getLogger(__name__)

class IssuesCrew:
    """
    Specialized crew for identifying, evaluating, and reporting issues in documents.
    Simplified implementation with standardized stage execution.
    """
    
    def __init__(
        self, 
        llm_client, 
        verbose=True, 
        max_chunk_size=1500,
        max_rpm=10,
        config_manager=None
    ):
        """
        Initialize the Issues Identification crew.
        
        Args:
            llm_client: LLM adapter
            verbose: Whether to enable verbose logging
            max_chunk_size: Maximum chunk size
            max_rpm: Maximum requests per minute
            config_manager: Optional config manager
        """
        # Store basic configuration
        self.llm_client = llm_client  
        self.verbose = verbose
        self.max_chunk_size = max_chunk_size
        self.max_rpm = max_rpm
        self.config_manager = config_manager
        
        # Get issues configuration (will be shared by all agents)
        self.config = None
        if config_manager:
            self.config = config_manager.get_config("issues")
        
        # Create document processing components
        self.document_analyzer = DocumentAnalyzer(llm_client)
        self.chunker = DocumentChunker()
        
        # Initialize the agent factory
        self._init_agent_factory()
        
        logger.info("IssuesCrew initialized with all agents")
    
    def _init_agent_factory(self):
        """Initialize the agent factory for creating agents on demand."""
        self.agents = {}
        
        # Define agent classes
        self.agent_classes = {
            "planner": PlannerAgent,
            "extractor": ExtractorAgent,
            "aggregator": AggregatorAgent,
            "evaluator": EvaluatorAgent,
            "formatter": FormatterAgent,
            "reviewer": ReviewerAgent
        }
    
    def _get_agent(self, agent_type: str) -> BaseAgent:
        """
        Get or create an agent by type.
        
        Args:
            agent_type: Type of agent to get
            
        Returns:
            Agent instance
        """
        if agent_type not in self.agents:
            if agent_type not in self.agent_classes:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            # Create the agent
            self.agents[agent_type] = self.agent_classes[agent_type](
                llm_client=self.llm_client,
                crew_type="issues",
                config=self.config,
                config_manager=self.config_manager,
                verbose=self.verbose,
                max_chunk_size=self.max_chunk_size,
                max_rpm=self.max_rpm
            )
        
        return self.agents[agent_type]
    
    async def process_document_with_context(self, context, progress_callback=None):
        """
        Process a document using the provided context.
        
        Args:
            context: ProcessingContext object
            progress_callback: Progress callback
            
        Returns:
            ProcessingContext with results
        """
        try:
            # Process through each stage
            await self._execute_stage(
                context, "document_analysis", self._analyze_document,
                0.05, "Analyzing document...", progress_callback
            )
            
            await self._execute_stage(
                context, "document_chunking", self._chunk_document,
                0.12, "Chunking document...", progress_callback
            )
            
            await self._execute_stage(
                context, "planning", self._create_plan,
                0.17, "Creating analysis plan...", progress_callback
            )
            
            await self._execute_stage(
                context, "extraction", self._extract_issues,
                0.22, "Extracting issues from document...", progress_callback
            )
            
            await self._execute_stage(
                context, "aggregation", self._aggregate_issues,
                0.55, "Aggregating issues...", progress_callback
            )
            
            await self._execute_stage(
                context, "evaluation", self._evaluate_issues,
                0.7, "Evaluating issues...", progress_callback
            )
            
            await self._execute_stage(
                context, "formatting", self._format_report,
                0.8, "Creating report...", progress_callback
            )
            
            # Only run review if enabled
            if context.options.get("enable_reviewer", True):
                await self._execute_stage(
                    context, "review", self._review_report,
                    0.9, "Reviewing report...", progress_callback
                )
            
            # Mark processing as complete
            if progress_callback:
                progress_callback(1.0, "Analysis complete")
            
            return context
            
        except Exception as e:
            logger.error(f"Error in issues analysis: {e}")
            
            # Update context with error
            if hasattr(context, 'metadata') and 'current_stage' in context.metadata:
                current_stage = context.metadata['current_stage']
                context.fail_stage(current_stage, str(e))
            
            # Re-raise the exception
            raise
    
    async def _execute_stage(self, context, stage_name, stage_method, progress_value, progress_message, progress_callback=None):
        """
        Execute a processing stage with standardized error handling and progress tracking.
        
        Args:
            context: ProcessingContext object
            stage_name: Name of the stage
            stage_method: Method to execute
            progress_value: Progress value (0.0 to 1.0)
            progress_message: Progress message
            progress_callback: Optional progress callback
        
        Returns:
            Stage result
        """
        context.set_stage(stage_name)
        if progress_callback:
            progress_callback(progress_value, progress_message)
        
        try:
            # Store progress callback in context for substage progress
            if not hasattr(context, 'metadata'):
                context.metadata = {}
            context.metadata['progress_callback'] = progress_callback
            
            # Execute the stage method
            result = await stage_method(context)
            
            # Mark stage as complete
            context.complete_stage(stage_name, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in {stage_name}: {e}")
            context.fail_stage(stage_name, str(e))
            raise
    
    async def _analyze_document(self, context):
        """
        Analyze document to extract metadata.
        
        Args:
            context: ProcessingContext object
            
        Returns:
            Document analysis results
        """
        # Get document text from context
        document_text = context.document_text
        
        # Analyze document
        document_info = await self.document_analyzer.analyze_preview(document_text)
        
        # Store analysis results in context
        context.document_info = document_info
        
        return document_info
    
    async def _chunk_document(self, context):
        """
        Chunk document for processing.
        
        Args:
            context: ProcessingContext object
            
        Returns:
            Chunking results
        """
        # Get document text and options
        document_text = context.document_text
        min_chunks = context.options.get("min_chunks", 3)
        max_chunk_size = context.options.get("max_chunk_size", self.max_chunk_size)
        
        # Chunk the document
        chunk_objects = self.chunker.chunk_document(
            document_text,
            min_chunks=min_chunks,
            max_chunk_size=max_chunk_size
        )
        
        # Store chunks and metadata in context
        context.chunks = [chunk["text"] for chunk in chunk_objects]
        context.chunk_metadata = chunk_objects
        
        return {
            "chunk_count": len(context.chunks),
            "chunk_metadata": chunk_objects
        }
    
    async def _create_plan(self, context):
        """
        Create a plan using the Planner agent.
        
        Args:
            context: ProcessingContext object
            
        Returns:
            Planning result
        """
        # Get the planner agent
        planner = self._get_agent("planner")
        
        # Create the plan
        plan = await planner.process(context)
        
        return plan
    
    async def _extract_issues(self, context):
        """
        Extract issues from document chunks.
        
        Args:
            context: ProcessingContext object
            
        Returns:
            Extraction results
        """
        # Get the extractor agent
        extractor = self._get_agent("extractor")
        
        # Extract issues
        extraction_results = await extractor.process(context)
        
        return extraction_results
    
    async def _aggregate_issues(self, context):
        """
        Aggregate issues from extraction results.
        
        Args:
            context: ProcessingContext object
            
        Returns:
            Aggregation results
        """
        # Get the aggregator agent
        aggregator = self._get_agent("aggregator")
        
        # Aggregate issues
        aggregated_result = await aggregator.process(context)
        
        return aggregated_result
    
    async def _evaluate_issues(self, context):
        """
        Evaluate aggregated issues.
        
        Args:
            context: ProcessingContext object
            
        Returns:
            Evaluation results
        """
        # Get the evaluator agent
        evaluator = self._get_agent("evaluator")
        
        # Evaluate issues
        evaluated_result = await evaluator.process(context)
        
        return evaluated_result
    
    async def _format_report(self, context):
        """
        Format the issues report.
        
        Args:
            context: ProcessingContext object
            
        Returns:
            Formatting results
        """
        # Get the formatter agent
        formatter = self._get_agent("formatter")
        
        # Format the report
        formatted_result = await formatter.process(context)
        
        return formatted_result
    
    async def _review_report(self, context):
        """
        Review the formatted report.
        
        Args:
            context: ProcessingContext object
            
        Returns:
            Review results
        """
        # Get the reviewer agent
        reviewer = self._get_agent("reviewer")
        
        # Review the report
        review_result = await reviewer.process(context)
        
        return review_result
    
    # Legacy method for backward compatibility
    async def process_document(
        self, 
        document_text,
        document_info=None, 
        user_preferences=None, 
        max_chunk_size=None,
        min_chunks=3,
        enable_reviewer=True,
        progress_callback=None
    ):
        """
        Legacy method for backward compatibility.
        Creates a context and uses process_document_with_context.
        
        Args:
            document_text: Document text to process
            document_info: Optional pre-loaded document info
            user_preferences: Optional user preferences
            max_chunk_size: Optional max chunk size
            min_chunks: Minimum number of chunks
            enable_reviewer: Whether to enable the reviewer
            progress_callback: Optional progress callback
            
        Returns:
            Processing results
        """
        # Import ProcessingContext
        from process_context import ProcessingContext
        
        # Create options dictionary
        options = {
            "detail_level": user_preferences.get("detail_level", "standard") if user_preferences else "standard",
            "focus_areas": user_preferences.get("focus_areas", []) if user_preferences else [],
            "user_instructions": user_preferences.get("user_instructions", "") if user_preferences else "",
            "max_chunk_size": max_chunk_size or self.max_chunk_size,
            "min_chunks": min_chunks,
            "enable_reviewer": enable_reviewer
        }
        
        # Create context
        context = ProcessingContext(document_text, options)
        
        # If document_info is provided, add it to context
        if document_info:
            context.document_info = document_info
        
        # Process with the new method
        await self.process_document_with_context(context, progress_callback)
        
        # Return results in the old format
        return context.get_final_result()
        
    # Synchronous version for easier integration
    def process_document_sync(
        self, 
        document_text,
        document_info=None, 
        user_preferences=None, 
        max_chunk_size=None,
        min_chunks=3,
        enable_reviewer=True,
        progress_callback=None
    ):
        """
        Synchronous version of process_document.
        
        Args:
            document_text: Document text to process
            document_info: Optional pre-loaded document info
            user_preferences: Optional user preferences
            max_chunk_size: Optional max chunk size
            min_chunks: Minimum number of chunks
            enable_reviewer: Whether to enable the reviewer
            progress_callback: Optional progress callback
            
        Returns:
            Processing results
        """
        # Create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async method in the event loop
            return loop.run_until_complete(
                self.process_document(
                    document_text=document_text,
                    document_info=document_info,
                    user_preferences=user_preferences,
                    max_chunk_size=max_chunk_size,
                    min_chunks=min_chunks,
                    enable_reviewer=enable_reviewer,
                    progress_callback=progress_callback
                )
            )
        finally:
            # Always close the loop
            loop.close()