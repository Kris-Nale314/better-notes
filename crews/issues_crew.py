"""
Enhanced Issues Crew - Specialized crew for identifying issues in documents.
Uses the updated architecture with UniversalLLMAdapter and ConfigManager.
"""

import logging
import time
import asyncio
import json
import os
import traceback
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

# Import our universal adapter and config manager
from universal_llm_adapter import UniversalLLMAdapter
from config_manager import ConfigManager

# Import agent components
from agents.extractor import ExtractorAgent
from agents.aggregator import AggregatorAgent
from agents.evaluator import EvaluatorAgent
from agents.formatter import FormatterAgent
from agents.reviewer import ReviewerAgent

logger = logging.getLogger(__name__)

class IssuesCrew:
    """
    Specialized crew for identifying, evaluating, and reporting issues in documents.
    Updated to work with the new architecture.
    """
    
    def __init__(
        self, 
        llm_client, 
        config_path=None, 
        verbose=True, 
        max_chunk_size=1500,
        max_rpm=10
    ):
        """
        Initialize the Issues Identification crew.
        
        Args:
            llm_client: Universal LLM adapter or compatible client
            config_path: Optional path to config file
            verbose: Whether to enable verbose logging
            max_chunk_size: Maximum chunk size for document processing
            max_rpm: Maximum requests per minute
        """
        # Ensure we have a UniversalLLMAdapter
        if not isinstance(llm_client, UniversalLLMAdapter):
            self.llm_client = UniversalLLMAdapter(llm_client=llm_client)
        else:
            self.llm_client = llm_client
            
        self.verbose = verbose
        self.max_chunk_size = max_chunk_size
        self.max_rpm = max_rpm
        
        # Create config manager
        self.config_manager = ConfigManager()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Track process state
        self.process_state = {"status": "initialized", "stages": {}, "errors": []}
        self.start_time = None
        self.run_id = None
        
        # Initialize specialized agents
        self.extractor_agent = ExtractorAgent(
            llm_client=self.llm_client,
            crew_type="issues",
            config=self.config,
            verbose=verbose,
            max_chunk_size=max_chunk_size,
            max_rpm=max_rpm
        )
        
        self.aggregator_agent = AggregatorAgent(
            llm_client=self.llm_client,
            crew_type="issues", 
            config=self.config,
            verbose=verbose,
            max_chunk_size=max_chunk_size,
            max_rpm=max_rpm
        )
        
        self.evaluator_agent = EvaluatorAgent(
            llm_client=self.llm_client,
            crew_type="issues",
            config=self.config,
            verbose=verbose,
            max_chunk_size=max_chunk_size,
            max_rpm=max_rpm
        )
        
        self.formatter_agent = FormatterAgent(
            llm_client=self.llm_client,
            crew_type="issues",
            config=self.config,
            verbose=verbose,
            max_chunk_size=max_chunk_size,
            max_rpm=max_rpm
        )
        
        self.reviewer_agent = ReviewerAgent(
            llm_client=self.llm_client,
            crew_type="issues",
            config=self.config,
            verbose=verbose,
            max_chunk_size=max_chunk_size,
            max_rpm=max_rpm
        )
    
    def update_rpm(self, new_rpm: int) -> None:
        """Update the maximum requests per minute for all agents."""
        self.max_rpm = new_rpm
        
        # Update all agents
        for agent_name in ["extractor_agent", "aggregator_agent", 
                          "evaluator_agent", "formatter_agent", "reviewer_agent"]:
            agent = getattr(self, agent_name)
            if hasattr(agent, "agent") and hasattr(agent.agent, "max_rpm"):
                agent.agent.max_rpm = new_rpm
    
    def _load_config(self, config_path=None):
        """Load the configuration file using ConfigManager."""
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    logger.info(f"Loaded configuration from {config_path}")
                    return config
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.error(f"Error loading config from {config_path}: {str(e)}")
        
        # Use ConfigManager to load the config
        return self.config_manager.get_config("issues")
    
    def process_document(
        self, 
        document_text_or_chunks,
        document_info: Optional[Dict[str, Any]] = None, 
        user_preferences: Optional[Dict[str, Any]] = None, 
        max_chunk_size: Optional[int] = None,
        min_chunks: int = 3,
        enable_reviewer: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process a document to identify issues with improved progress reporting.
        
        Args:
            document_text_or_chunks: Document text or pre-chunked document
            document_info: Document metadata
            user_preferences: User preferences
            max_chunk_size: Maximum chunk size
            min_chunks: Minimum number of chunks
            enable_reviewer: Whether to enable the reviewer step
            progress_callback: Callback for progress updates
            
        Returns:
            Analysis results
        """
        # Initialize run tracking
        self.start_time = time.time()
        self.run_id = f"issues-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Store initial process state
        self.process_state = {
            "status": "started",
            "run_id": self.run_id,
            "start_time": self.start_time,
            "enable_reviewer": enable_reviewer,
            "stages": {},
            "errors": []
        }
        
        # Update max chunk size if provided
        if max_chunk_size is not None:
            self.max_chunk_size = max_chunk_size
        
        try:
            # Stage 1: Prepare document and chunks
            self._start_stage("document_preparation")
            if progress_callback:
                progress_callback(0.05, "Preparing document...")
            
            chunks, chunk_metadata = self._prepare_document(
                document_text_or_chunks, 
                document_info, 
                user_preferences,
                min_chunks
            )
            
            self.process_state["document_length"] = len(document_text_or_chunks) if isinstance(document_text_or_chunks, str) else sum(len(c) for c in document_text_or_chunks)
            self.process_state["chunk_count"] = len(chunks)
            self._complete_stage("document_preparation")
            
            # Stage 2: Apply agent instructions from planner
            if progress_callback:
                progress_callback(0.1, "Setting up analysis pipeline...")
                
            # Apply any custom instructions to agents
            custom_instructions = user_preferences.get("agent_instructions", {})
            self._apply_agent_instructions(custom_instructions)
            
            # Stage 3: Extract issues from chunks in parallel
            self._start_stage("extraction")
            if progress_callback:
                progress_callback(0.15, "Extracting issues from document...")
            
            extraction_results = self._extract_from_chunks(chunks, chunk_metadata, document_info, progress_callback)
            
            # Store extraction metadata
            self.process_state["stages"]["extraction"]["results"] = {
                "chunk_count": len(chunks),
                "successful_chunks": sum(1 for r in extraction_results if isinstance(r, dict) and "error" not in r),
                "issue_count": sum(len(r.get("issues", [])) for r in extraction_results if isinstance(r, dict) and "issues" in r)
            }
            
            self._complete_stage("extraction")
            
            # Stage 4: Aggregation
            self._start_stage("aggregation")
            if progress_callback:
                progress_callback(0.55, "Aggregating and deduplicating issues...")
            
            aggregated_result = self._aggregate_results(extraction_results, document_info)
            self._complete_stage("aggregation")
            
            # Stage 5: Evaluation
            self._start_stage("evaluation")
            if progress_callback:
                progress_callback(0.65, "Evaluating issue severity and impact...")
            
            evaluated_result = self._evaluate_results(aggregated_result, document_info)
            self._complete_stage("evaluation")
            
            # Stage 6: Formatting
            self._start_stage("formatting")
            if progress_callback:
                progress_callback(0.75, "Creating structured report...")
            
            formatted_result = self._format_results(evaluated_result, document_info, user_preferences)
            self._complete_stage("formatting")
            
            # Stage 7: Review (optional)
            final_result = formatted_result
            if enable_reviewer:
                self._start_stage("review")
                if progress_callback:
                    progress_callback(0.85, "Performing quality review...")
                
                review_result = self._review_results(formatted_result, document_info, user_preferences)
                
                # Combine formatted result with review
                final_result = {
                    "raw_output": formatted_result,
                    "review_result": review_result,
                }
                self._complete_stage("review")
            
            # Record processing time
            end_time = time.time()
            duration = end_time - self.start_time
            
            self.process_state["status"] = "completed"
            self.process_state["end_time"] = end_time
            self.process_state["duration"] = duration
            
            if progress_callback:
                progress_callback(1.0, f"Analysis complete")
            
            # Add metadata
            if isinstance(final_result, dict):
                final_result["_metadata"] = {
                    "run_id": self.run_id,
                    "duration": duration,
                    "document_info": document_info if document_info else {},
                    "processing_stats": self.process_state
                }
            
            return final_result
            
        except Exception as e:
            self._fail_stage(self.process_state.get("current_stage", "unknown"), str(e))
            logger.error(f"Error in issues analysis: {str(e)}", exc_info=True)
            if progress_callback:
                progress_callback(1.0, f"Error: {str(e)}")
            
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "_metadata": {
                    "status": "error",
                    "run_id": self.run_id,
                    "duration": time.time() - self.start_time,
                    "processing_stats": self.process_state
                }
            }
    
    def _prepare_document(
        self, 
        document_text_or_chunks, 
        document_info, 
        user_preferences,
        min_chunks
    ):
        """Prepare document by chunking or using provided chunks."""
        chunk_metadata = []
        
        if isinstance(document_text_or_chunks, str):
            # We have full text - chunk it
            try:
                from lean.chunker import DocumentChunker
                chunker = DocumentChunker()
            except ImportError:
                try:
                    from chunker import DocumentChunker
                    chunker = DocumentChunker()
                except ImportError:
                    logger.error("Could not import DocumentChunker")
                    # Fallback to simple chunking
                    return self._simple_chunk_text(document_text_or_chunks, min_chunks), chunk_metadata
            
            # Calculate appropriate chunk size
            if self.max_chunk_size is None:
                doc_length = len(document_text_or_chunks)
                calculated_chunk_size = doc_length // min_chunks
                self.max_chunk_size = min(calculated_chunk_size + 100, 16000)
            
            try:
                chunk_objects = chunker.chunk_document(
                    document_text_or_chunks,
                    min_chunks=min_chunks,
                    max_chunk_size=self.max_chunk_size
                )
                
                # Extract text and metadata
                chunks = []
                for chunk in chunk_objects:
                    chunks.append(chunk["text"])
                    chunk_metadata.append({
                        "position": chunk.get("position", ""),
                        "chunk_type": chunk.get("chunk_type", ""),
                        "index": chunk.get("index", 0)
                    })
            except Exception as e:
                logger.error(f"Error in document chunking: {e}")
                # Fallback to simple chunking
                return self._simple_chunk_text(document_text_or_chunks, min_chunks), chunk_metadata
            
            # Analyze document if needed
            if not document_info:
                try:
                    from lean.document import DocumentAnalyzer
                    analyzer = DocumentAnalyzer(self.llm_client)
                    document_info = asyncio.run(analyzer.analyze_preview(document_text_or_chunks))
                    document_info["original_text_length"] = len(document_text_or_chunks)
                except Exception as e:
                    logger.warning(f"Document analysis failed: {e}")
                    # Create basic document_info
                    document_info = {
                        "original_text_length": len(document_text_or_chunks),
                        "basic_stats": {
                            "word_count": len(document_text_or_chunks.split()),
                            "char_count": len(document_text_or_chunks)
                        }
                    }
        
        elif isinstance(document_text_or_chunks, list):
            # We already have chunks
            chunks = document_text_or_chunks
            chunk_metadata = [{"index": i} for i in range(len(chunks))]
        else:
            raise TypeError("Expected either a string or a list of chunks")
        
        return chunks, chunk_metadata
    
    def _simple_chunk_text(self, text, min_chunks):
        """Simple fallback chunking for when DocumentChunker is unavailable."""
        chunk_size = len(text) // min_chunks
        chunks = []
        
        # Create chunks of roughly equal size
        for i in range(min_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < min_chunks - 1 else len(text)
            chunks.append(text[start:end])
            
        return chunks
    
    def _apply_agent_instructions(self, custom_instructions):
        """Apply custom instructions to agents."""
        for agent_type, instructions in custom_instructions.items():
            # Map agent type to agent attribute
            agent_map = {
                "extraction": "extractor_agent",
                "aggregation": "aggregator_agent", 
                "evaluation": "evaluator_agent",
                "formatting": "formatter_agent",
                "reviewer": "reviewer_agent"
            }
            
            agent_attr = agent_map.get(agent_type)
            if agent_attr and hasattr(self, agent_attr):
                # Set custom instructions on the agent
                agent_obj = getattr(self, agent_attr)
                agent_obj.custom_instructions = instructions
                
                if self.verbose:
                    logger.info(f"Applied custom instructions to {agent_type} agent")
    
    def _extract_from_chunks(self, chunks, chunk_metadata, document_info, progress_callback):
        """
        Extract issues from chunks with improved progress reporting.
        
        Args:
            chunks: Document chunks
            chunk_metadata: Metadata for each chunk
            document_info: Document metadata
            progress_callback: Progress callback
            
        Returns:
            Extraction results
        """
        async def extract_all_chunks():
            # Set up concurrency control
            max_concurrency = min(len(chunks), 10)
            semaphore = asyncio.Semaphore(max_concurrency)
            
            async def process_chunk(idx):
                async with semaphore:
                    chunk = chunks[idx]
                    metadata = chunk_metadata[idx] if idx < len(chunk_metadata) else {}
                    
                    # Calculate progress for this chunk within the extraction phase
                    # Extraction is from 0.15 to 0.55 of total progress
                    chunk_progress = 0.15 + (0.4 * (idx / len(chunks)))
                    
                    # Only log at verbose level - don't send to UI
                    if self.verbose:
                        logger.info(f"Processing chunk {idx+1}/{len(chunks)}")
                    
                    # Only update progress every few chunks to avoid UI spam
                    if progress_callback and idx % max(1, len(chunks)//5) == 0:
                        progress_callback(chunk_progress, f"Extracting issues ({idx+1}/{len(chunks)} chunks)")
                    
                    try:
                        # Extract issues from this chunk
                        extraction_context = {
                            "document_chunk": chunk,
                            "document_info": document_info,
                            "chunk_metadata": metadata
                        }
                        
                        # Execute the task with robust error handling
                        start_time = time.time()
                        result = self.extractor_agent.execute_task(context=extraction_context)
                        duration = time.time() - start_time
                        
                        # Validate result format 
                        if isinstance(result, str):
                            try:
                                # Try to parse as JSON
                                result = json.loads(result)
                            except json.JSONDecodeError:
                                # Keep as string for later handling
                                pass
                        
                        # Add processing metadata if result is a dict
                        if isinstance(result, dict):
                            result["_metadata"] = {
                                "chunk_index": idx,
                                "processing_time": duration,
                                "timestamp": datetime.now().isoformat()
                            }
                        
                        return result
                    except Exception as e:
                        logger.error(f"Error processing chunk {idx}: {str(e)}")
                        return {
                            "error": str(e), 
                            "chunk_index": idx,
                            "_metadata": {
                                "error": True,
                                "timestamp": datetime.now().isoformat()
                            }
                        }
            
            # Create tasks for all chunks
            tasks = [process_chunk(i) for i in range(len(chunks))]
            
            # Execute all tasks
            results = await asyncio.gather(*tasks)
            
            # Sort results by chunk index
            return sorted(results, key=lambda r: r.get("_metadata", {}).get("chunk_index", 0) if isinstance(r, dict) else 0)
        
        # Run the extraction process
        try:
            # Check if there's an event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            return loop.run_until_complete(extract_all_chunks())
        except Exception as e:
            logger.error(f"Error in parallel extraction: {str(e)}")
            if progress_callback:
                progress_callback(0.4, f"Error in extraction: {str(e)}")
            raise
    
    def _aggregate_results(self, extraction_results, document_info):
        """Aggregate extraction results."""
        try:
            # Build context for aggregation
            context = {
                "extraction_results": extraction_results,
                "document_info": document_info or {}
            }
            
            # Calculate stats for prompt context
            successful_extractions = sum(1 for r in extraction_results if isinstance(r, dict) and "error" not in r)
            total_issues = sum(len(r.get("issues", [])) for r in extraction_results 
                             if isinstance(r, dict) and "issues" in r)
            
            context["extraction_stats"] = {
                "total_chunks": len(extraction_results),
                "successful_chunks": successful_extractions,
                "total_issues": total_issues
            }
            
            # Execute aggregation
            result = self.aggregator_agent.execute_task(context=context)
            return result
        except Exception as e:
            logger.error(f"Error in aggregation: {str(e)}")
            raise
    
    def _evaluate_results(self, aggregated_result, document_info):
        """Evaluate aggregated results."""
        try:
            # Build context for evaluation
            context = {
                "aggregated_result": aggregated_result,
                "document_info": document_info or {}
            }
            
            # Execute evaluation
            result = self.evaluator_agent.execute_task(context=context)
            return result
        except Exception as e:
            logger.error(f"Error in evaluation: {str(e)}")
            raise
    
    def _format_results(self, evaluated_result, document_info, user_preferences):
        """Format evaluated results."""
        try:
            # Build context for formatting
            context = {
                "evaluated_result": evaluated_result,
                "document_info": document_info or {},
                "user_preferences": user_preferences or {}
            }
            
            # Execute formatting
            result = self.formatter_agent.execute_task(context=context)
            return result
        except Exception as e:
            logger.error(f"Error in formatting: {str(e)}")
            raise
    
    def _review_results(self, formatted_result, document_info, user_preferences):
        """Review formatted results."""
        try:
            # Build context for review
            context = {
                "formatted_result": formatted_result,
                "document_info": document_info or {},
                "user_preferences": user_preferences or {}
            }
            
            # Execute review
            result = self.reviewer_agent.execute_task(context=context)
            return result
        except Exception as e:
            logger.error(f"Error in review: {str(e)}")
            raise
    
    def _start_stage(self, stage_name: str) -> None:
        """Begin tracking a processing stage."""
        self.process_state["current_stage"] = stage_name
        self.process_state["stages"][stage_name] = {
            "status": "started",
            "start_time": time.time()
        }
    
    def _complete_stage(self, stage_name: str) -> None:
        """Mark a processing stage as completed."""
        stage = self.process_state["stages"].get(stage_name, {})
        stage["status"] = "completed"
        stage["end_time"] = time.time()
        stage["duration"] = round(stage["end_time"] - stage.get("start_time", self.start_time), 2)
        self.process_state["stages"][stage_name] = stage
    
    def _fail_stage(self, stage_name: str, error_message: str) -> None:
        """Mark a stage as failed and log the error."""
        stage = self.process_state["stages"].get(stage_name, {})
        stage["status"] = "failed"
        stage["end_time"] = time.time()
        stage["error"] = error_message
        stage["duration"] = round(stage["end_time"] - stage.get("start_time", self.start_time), 2)
        self.process_state["stages"][stage_name] = stage
        
        self.process_state["errors"].append({
            "stage": stage_name,
            "message": error_message,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.error(f"Stage '{stage_name}' failed: {error_message}")