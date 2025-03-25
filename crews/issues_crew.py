# crews/issues_crew.py
"""
Enhanced Issues Crew with integrated Planner and ProcessingContext support.
Analyzes documents to identify issues, problems, risks, and challenges.
"""

import logging
import asyncio
import time
import json
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
    Contains all required agents, including the Planner, and works with ProcessingContext.
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
        
        # Initialize all agents (including Planner)
        self._init_agents()
        
        logger.info("IssuesCrew initialized with all agents")
    
    def _init_agents(self):
        """Initialize all agents needed for the crew."""
        try:
            # Create Planner agent (now part of the crew)
            self.planner_agent = PlannerAgent(
                llm_client=self.llm_client,
                crew_type="issues",
                config=self.config,
                config_manager=self.config_manager,
                verbose=self.verbose,
                max_chunk_size=self.max_chunk_size,
                max_rpm=self.max_rpm
            )
            
            # Create other specialized agents
            self.extractor_agent = ExtractorAgent(
                llm_client=self.llm_client,
                crew_type="issues",
                config=self.config,
                config_manager=self.config_manager,
                verbose=self.verbose,
                max_chunk_size=self.max_chunk_size,
                max_rpm=self.max_rpm
            )
            
            self.aggregator_agent = AggregatorAgent(
                llm_client=self.llm_client,
                crew_type="issues", 
                config=self.config,
                config_manager=self.config_manager,
                verbose=self.verbose,
                max_chunk_size=self.max_chunk_size,
                max_rpm=self.max_rpm
            )
            
            self.evaluator_agent = EvaluatorAgent(
                llm_client=self.llm_client,
                crew_type="issues",
                config=self.config,
                config_manager=self.config_manager,
                verbose=self.verbose,
                max_chunk_size=self.max_chunk_size,
                max_rpm=self.max_rpm
            )
            
            self.formatter_agent = FormatterAgent(
                llm_client=self.llm_client,
                crew_type="issues",
                config=self.config,
                config_manager=self.config_manager,
                verbose=self.verbose,
                max_chunk_size=self.max_chunk_size,
                max_rpm=self.max_rpm
            )
            
            self.reviewer_agent = ReviewerAgent(
                llm_client=self.llm_client,
                crew_type="issues",
                config=self.config,
                config_manager=self.config_manager,
                verbose=self.verbose,
                max_chunk_size=self.max_chunk_size,
                max_rpm=self.max_rpm
            )
            
            logger.info("All agents initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
            raise
    
    # crews/issues_crew.py
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
            # Stage 1: Document Analysis
            await self._analyze_document(context, progress_callback)
            
            # Stage 2: Document Chunking
            self._chunk_document(context, progress_callback)
            
            # Stage 3: Planning
            await self._create_plan(context, progress_callback)
            
            # Stage 4: Extraction
            await self._extract_issues(context, progress_callback)
            
            # Stage 5: Aggregation
            await self._aggregate_issues(context, progress_callback)
            
            # Stage 6: Evaluation
            await self._evaluate_issues(context, progress_callback)
            
            # Stage 7: Formatting
            await self._format_report(context, progress_callback)
            
            # Stage 8: Review (optional)
            if context.options.get("enable_reviewer", True):
                await self._review_report(context, progress_callback)
            
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
            
            # Re-raise the exception to be handled by the Orchestrator
            raise

    # 1. _analyze_document
    async def _analyze_document(self, context, progress_callback):
        """Analyze the document to extract metadata."""
        context.set_stage("document_analysis")
        if progress_callback:
            progress_callback(0.05, "Analyzing document...")
        
        try:
            # Get document text from context
            document_text = context.document_text
            
            # Analyze document to get metadata
            document_info = await self.document_analyzer.analyze_preview(document_text)
            
            # Store analysis results in context
            context.document_info = document_info
            
            # Mark stage as complete
            context.complete_stage("document_analysis", document_info)
            
            if progress_callback:
                progress_callback(0.1, "Document analysis complete")
                
        except Exception as e:
            logger.error(f"Error in document analysis: {e}")
            context.fail_stage("document_analysis", str(e))
            raise

    def _chunk_document(self, context, progress_callback):
        """Chunk the document for processing."""
        context.set_stage("document_chunking")
        if progress_callback:
            progress_callback(0.12, "Chunking document...")
        
        try:
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
            
            # Mark stage as complete
            context.complete_stage("document_chunking", {
                "chunk_count": len(context.chunks),
                "chunk_objects": chunk_objects
            })
            
            if progress_callback:
                progress_callback(0.15, f"Document chunked into {len(context.chunks)} sections")
                
        except Exception as e:
            logger.error(f"Error in document chunking: {e}")
            context.fail_stage("document_chunking", str(e))
            raise

    # 2. _create_plan
    async def _create_plan(self, context, progress_callback):
        """Create a plan using the Planner agent."""
        context.set_stage("planning")
        if progress_callback:
            progress_callback(0.17, "Creating analysis plan...")
        
        try:
            # Prepare the planning context
            user_preferences = {
                "detail_level": context.options.get("detail_level", "standard"),
                "focus_areas": context.options.get("focus_areas", []),
                "user_instructions": context.options.get("user_instructions", "")
            }
            
            # Let the Planner create a plan
            plan = await self.planner_agent.create_plan(
                document_info=context.document_info,
                user_preferences=user_preferences,
                crew_type="issues"
            )
            
            # Store plan in context
            context.agent_instructions = plan
            
            # Mark stage as complete
            context.complete_stage("planning", plan)
            
            if progress_callback:
                progress_callback(0.2, "Analysis plan created")
                
        except Exception as e:
            logger.error(f"Error in planning: {e}")
            context.fail_stage("planning", str(e))
            raise

    # 3. _extract_issues
    async def _extract_issues(self, context, progress_callback):
        """Extract issues from document chunks."""
        context.set_stage("extraction")
        if progress_callback:
            progress_callback(0.22, "Extracting issues from document...")
        
        try:
            # Process each chunk to extract issues
            extraction_results = []
            chunks = context.chunks
            chunk_metadata = context.chunk_metadata
            
            # Track extraction progress
            extraction_progress_range = (0.22, 0.5)  # 22% to 50% of total progress
            progress_per_chunk = (extraction_progress_range[1] - extraction_progress_range[0]) / len(chunks)
            
            for i, chunk in enumerate(chunks):
                # Update progress for this chunk
                if progress_callback:
                    chunk_progress = extraction_progress_range[0] + (i * progress_per_chunk)
                    progress_callback(
                        chunk_progress, 
                        f"Extracting issues from chunk {i+1}/{len(chunks)}"
                    )
                
                # Get metadata for this chunk
                metadata = chunk_metadata[i] if i < len(chunk_metadata) else {"index": i}
                
                # Extract issues from this chunk
                result = await self.extractor_agent.extract_from_chunk(
                    chunk=chunk,
                    document_info=context.document_info,
                    chunk_metadata=metadata
                )
                
                # Add to results
                extraction_results.append(result)
            
            # Store extraction results in context
            context.results["extraction"] = extraction_results
            
            # Mark stage as complete
            context.complete_stage("extraction", extraction_results)
            
            if progress_callback:
                progress_callback(0.5, "Issue extraction complete")
                
        except Exception as e:
            logger.error(f"Error in extraction: {e}")
            context.fail_stage("extraction", str(e))
            raise

    # 4. _aggregate_issues
    async def _aggregate_issues(self, context, progress_callback):
        """Aggregate issues from multiple chunks."""
        context.set_stage("aggregation")
        if progress_callback:
            progress_callback(0.55, "Aggregating issues...")
        
        try:
            # Get extraction results
            extraction_results = context.results.get("extraction", [])
            
            # Aggregate issues
            aggregated_result = await self.aggregator_agent.aggregate_results(
                extraction_results=extraction_results,
                document_info=context.document_info
            )
            
            # Store aggregation results in context
            context.results["aggregation"] = aggregated_result
            
            # Mark stage as complete
            context.complete_stage("aggregation", aggregated_result)
            
            if progress_callback:
                progress_callback(0.65, "Issues aggregated")
                
        except Exception as e:
            logger.error(f"Error in aggregation: {e}")
            context.fail_stage("aggregation", str(e))
            raise

    # 5. _evaluate_issues
    async def _evaluate_issues(self, context, progress_callback):
        """Evaluate aggregated issues."""
        context.set_stage("evaluation")
        if progress_callback:
            progress_callback(0.7, "Evaluating issues...")
        
        try:
            # Get aggregation results
            aggregated_result = context.results.get("aggregation", {})
            
            # Evaluate issues
            evaluated_result = await self.evaluator_agent.evaluate_items(
                aggregated_items=aggregated_result,
                document_info=context.document_info
            )
            
            # Store evaluation results in context
            context.results["evaluation"] = evaluated_result
            
            # Mark stage as complete
            context.complete_stage("evaluation", evaluated_result)
            
            if progress_callback:
                progress_callback(0.75, "Issues evaluated")
                
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            context.fail_stage("evaluation", str(e))
            raise

    # 6. _format_report
    async def _format_report(self, context, progress_callback):
        """Format the issues report."""
        context.set_stage("formatting")
        if progress_callback:
            progress_callback(0.8, "Creating report...")
        
        try:
            # Get evaluation results
            evaluated_result = context.results.get("evaluation", {})
            
            # Format the report
            formatted_result = await self.formatter_agent.format_report(
                evaluated_items=evaluated_result,
                document_info=context.document_info,
                user_preferences=context.options
            )
            
            # Store formatting results in context
            context.results["formatting"] = formatted_result
            
            # Mark stage as complete
            context.complete_stage("formatting", formatted_result)
            
            if progress_callback:
                progress_callback(0.85, "Report created")
                
        except Exception as e:
            logger.error(f"Error in formatting: {e}")
            context.fail_stage("formatting", str(e))
            raise

    # 7. _review_report
    async def _review_report(self, context, progress_callback):
        """Review the formatted report."""
        context.set_stage("review")
        if progress_callback:
            progress_callback(0.9, "Reviewing report...")
        
        try:
            # Get formatted result
            formatted_result = context.results.get("formatting", {})
            
            # Review the report
            review_result = await self.reviewer_agent.review_analysis(
                formatted_result=formatted_result,
                document_info=context.document_info,
                user_preferences=context.options
            )
            
            # Store review results in context
            context.results["review"] = review_result
            
            # Mark stage as complete
            context.complete_stage("review", review_result)
            
            if progress_callback:
                progress_callback(0.95, "Report reviewed")
                
        except Exception as e:
            logger.error(f"Error in review: {e}")
            context.fail_stage("review", str(e))
            raise
    
    # Legacy method for backward compatibility
    def process_document(
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
        """
        # Import ProcessingContext
        from orchestrator import ProcessingContext
        
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
        self.process_document_with_context(context, progress_callback)
        
        # Return results in the old format
        return context.get_final_result()