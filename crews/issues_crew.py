"""
Enhanced Issues Crew - Specialized crew for identifying issues in documents.
Streamlined implementation that works with UniversalLLMAdapter and modern agent architecture.
"""

import logging
import time
import asyncio
import json
import os
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

# Import document processing components
from lean.chunker import DocumentChunker

logger = logging.getLogger(__name__)

class IssuesCrew:
    """
    Specialized crew for identifying, evaluating, and reporting issues in documents.
    Simplified implementation focusing on reliability.
    """
    
    def __init__(
        self, 
        llm_client, 
        verbose=True, 
        max_chunk_size=1500,
        max_rpm=10
    ):
        """
        Initialize the Issues Identification crew.
        
        Args:
            llm_client: Universal LLM adapter or compatible client
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
        self.config = self.config_manager.get_config("issues")
        if not self.config:
            logger.warning("Issues configuration not found. Using default configuration.")
            self.config = self._get_default_config()
        
        # Create document chunker
        self.chunker = DocumentChunker()
        
        # Track process state
        self.process_state = {"status": "initialized", "stages": {}, "errors": []}
        self.start_time = None
        self.run_id = None
        
        # Initialize specialized agents
        self._init_agents()
    
    def _init_agents(self):
        """Initialize all agents needed for the crew."""
        try:
            self.extractor_agent = ExtractorAgent(
                llm_client=self.llm_client,
                crew_type="issues",
                config=self.config,
                verbose=self.verbose,
                max_chunk_size=self.max_chunk_size,
                max_rpm=self.max_rpm
            )
            
            self.aggregator_agent = AggregatorAgent(
                llm_client=self.llm_client,
                crew_type="issues", 
                config=self.config,
                verbose=self.verbose,
                max_chunk_size=self.max_chunk_size,
                max_rpm=self.max_rpm
            )
            
            self.evaluator_agent = EvaluatorAgent(
                llm_client=self.llm_client,
                crew_type="issues",
                config=self.config,
                verbose=self.verbose,
                max_chunk_size=self.max_chunk_size,
                max_rpm=self.max_rpm
            )
            
            self.formatter_agent = FormatterAgent(
                llm_client=self.llm_client,
                crew_type="issues",
                config=self.config,
                verbose=self.verbose,
                max_chunk_size=self.max_chunk_size,
                max_rpm=self.max_rpm
            )
            
            self.reviewer_agent = ReviewerAgent(
                llm_client=self.llm_client,
                crew_type="issues",
                config=self.config,
                verbose=self.verbose,
                max_chunk_size=self.max_chunk_size,
                max_rpm=self.max_rpm
            )
            
            logger.info("All agents initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
            raise
    
    def update_rpm(self, new_rpm: int) -> None:
        """Update the maximum requests per minute for all agents."""
        self.max_rpm = new_rpm
        
        # Update all agents
        for agent_name in ["extractor_agent", "aggregator_agent", 
                          "evaluator_agent", "formatter_agent", "reviewer_agent"]:
            agent = getattr(self, agent_name)
            agent.max_rpm = new_rpm
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for issues analysis."""
        return {
            "metadata": {
                "version": "1.0",
                "description": "Default issues identification configuration"
            },
            "analysis_definition": {
                "issue": {
                    "definition": "Any problem, challenge, risk, or concern that may impact objectives, efficiency, or quality"
                },
                "severity_levels": {
                    "critical": "Immediate threat to operations, security, or compliance; blocks major deliverables",
                    "high": "Significant impact on effectiveness or efficiency; requires attention soon",
                    "medium": "Causes inefficiency or limitations; should be addressed",
                    "low": "Minor concern with minimal impact; could be addressed through regular improvements"
                }
            },
            "agents": {
                "extractor": {
                    "role": "Issue Extractor",
                    "goal": "Identify potential issues in document chunks",
                    "instructions": "Analyze the document to identify issues, problems, and challenges."
                },
                "aggregator": {
                    "role": "Issue Aggregator",
                    "goal": "Combine and deduplicate issues from multiple extractions",
                    "instructions": "Combine similar issues while preserving important distinctions."
                },
                "evaluator": {
                    "role": "Issue Evaluator",
                    "goal": "Assess severity and impact of identified issues",
                    "instructions": "Evaluate each issue for severity, impact, and priority."
                },
                "formatter": {
                    "role": "Report Formatter",
                    "goal": "Create a clear, structured report of issues",
                    "instructions": "Format the issues into a well-organized report grouped by severity."
                },
                "reviewer": {
                    "role": "Analysis Reviewer",
                    "goal": "Ensure analysis quality and alignment with user needs",
                    "instructions": "Review the report for quality, consistency, and completeness."
                }
            }
        }
    
    def process_document(
        self, 
        document_text,
        document_info: Optional[Dict[str, Any]] = None, 
        user_preferences: Optional[Dict[str, Any]] = None, 
        max_chunk_size: Optional[int] = None,
        min_chunks: int = 3,
        enable_reviewer: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process a document to identify issues.
        
        Args:
            document_text: Document text
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
        
        # Ensure we have user preferences
        if user_preferences is None:
            user_preferences = {}
        
        try:
            # Stage 1: Prepare document and chunks
            self._start_stage("document_preparation")
            if progress_callback:
                progress_callback(0.05, "Preparing document...")
            
            # Ensure document_text is a string
            if not isinstance(document_text, str):
                document_text = str(document_text)
            
            # Chunk the document
            chunks = self._chunk_document(document_text, min_chunks)
            chunk_metadata = [{"index": i, "position": f"chunk_{i}"} for i in range(len(chunks))]
            
            # Update state
            self.process_state["document_length"] = len(document_text)
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
                    "formatted_report": formatted_result,
                    "review_result": review_result
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
            final_result["_metadata"] = {
                "run_id": self.run_id,
                "duration": duration,
                "document_info": document_info if document_info else {},
                "processing_stats": self.process_state,
                "user_preferences": user_preferences
            }
            
            return final_result
            
        except Exception as e:
            self._fail_stage(self.process_state.get("current_stage", "unknown"), str(e))
            logger.error(f"Error in issues analysis: {e}")
            if progress_callback:
                progress_callback(1.0, f"Error: {str(e)}")
            
            return {
                "error": str(e),
                "_metadata": {
                    "status": "error",
                    "run_id": self.run_id,
                    "duration": time.time() - self.start_time,
                    "processing_stats": self.process_state
                }
            }
    
    def _chunk_document(self, document_text: str, min_chunks: int) -> List[str]:
        """
        Chunk document text using DocumentChunker.
        
        Args:
            document_text: Document text
            min_chunks: Minimum number of chunks
            
        Returns:
            List of text chunks
        """
        try:
            # Calculate appropriate chunk size
            if self.max_chunk_size is None:
                doc_length = len(document_text)
                max_chunk_size = doc_length // min_chunks
                self.max_chunk_size = min(max_chunk_size + 100, 16000)
            
            # Use the chunker
            chunk_objects = self.chunker.chunk_document(
                document_text,
                min_chunks=min_chunks,
                max_chunk_size=self.max_chunk_size
            )
            
            # Extract text from chunk objects
            chunks = [chunk["text"] for chunk in chunk_objects]
            
            return chunks
        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            # Fallback to simple chunking
            return self._simple_chunk_text(document_text, min_chunks)
    
    def _simple_chunk_text(self, text: str, min_chunks: int) -> List[str]:
        """Simple fallback chunking for when DocumentChunker fails."""
        chunk_size = len(text) // min_chunks
        chunks = []
        
        # Create chunks of roughly equal size
        for i in range(min_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < min_chunks - 1 else len(text)
            chunks.append(text[start:end])
            
        return chunks
    
    def _apply_agent_instructions(self, custom_instructions: Dict[str, Any]) -> None:
        """
        Apply custom instructions to agents.
        
        Args:
            custom_instructions: Custom instructions by agent type
        """
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
    
    def _extract_from_chunks(
        self, 
        chunks: List[str], 
        chunk_metadata: List[Dict[str, Any]], 
        document_info: Optional[Dict[str, Any]],
        progress_callback: Optional[Callable]
    ) -> List[Dict[str, Any]]:
        """
        Extract issues from chunks sequentially.
        
        Args:
            chunks: Document chunks
            chunk_metadata: Metadata for each chunk
            document_info: Document metadata
            progress_callback: Progress callback
            
        Returns:
            Extraction results
        """
        results = []
        extraction_progress_range = (0.15, 0.55)  # Progress range for extraction phase
        progress_per_chunk = (extraction_progress_range[1] - extraction_progress_range[0]) / len(chunks)
        
        for i, chunk in enumerate(chunks):
            # Calculate progress for this chunk
            if progress_callback:
                chunk_progress = extraction_progress_range[0] + (i * progress_per_chunk)
                progress_callback(chunk_progress, f"Extracting issues from chunk {i+1}/{len(chunks)}")
            
            # Get metadata for this chunk
            metadata = chunk_metadata[i] if i < len(chunk_metadata) else {"index": i}
            
            try:
                # Extract issues from this chunk
                extraction_context = {
                    "document_chunk": chunk,
                    "document_info": document_info or {},
                    "chunk_metadata": metadata
                }
                
                # Execute the extraction task
                start_time = time.time()
                result = self.extractor_agent.execute_task(context=extraction_context)
                duration = time.time() - start_time
                
                # Parse result if it's a string
                if isinstance(result, str):
                    try:
                        # Try to parse as JSON
                        result = self._parse_json_safely(result)
                    except:
                        # Keep as string if parsing fails
                        result = {"issues": [{"description": result, "chunk_index": i}]}
                
                # Add metadata if result is a dict
                if isinstance(result, dict):
                    result["_metadata"] = {
                        "chunk_index": i,
                        "processing_time": duration,
                        "timestamp": datetime.now().isoformat()
                    }
                
                # Add to results
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                results.append({
                    "error": str(e), 
                    "chunk_index": i,
                    "_metadata": {
                        "error": True,
                        "timestamp": datetime.now().isoformat()
                    }
                })
        
        return results
    
    def _aggregate_results(
        self, 
        extraction_results: List[Dict[str, Any]], 
        document_info: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate extraction results.
        
        Args:
            extraction_results: List of extraction results
            document_info: Document metadata
            
        Returns:
            Aggregated results
        """
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
        
        # Parse result if it's a string
        if isinstance(result, str):
            try:
                result = self._parse_json_safely(result)
            except:
                # If parsing fails, wrap in basic structure
                result = {
                    "aggregated_issues": [{"description": result}]
                }
        
        return result
    
    def _evaluate_results(
        self, 
        aggregated_result: Dict[str, Any], 
        document_info: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate aggregated results.
        
        Args:
            aggregated_result: Aggregated results
            document_info: Document metadata
            
        Returns:
            Evaluated results
        """
        # Build context for evaluation
        context = {
            "aggregated_result": aggregated_result,
            "document_info": document_info or {}
        }
        
        # Execute evaluation
        result = self.evaluator_agent.execute_task(context=context)
        
        # Parse result if it's a string
        if isinstance(result, str):
            try:
                result = self._parse_json_safely(result)
            except:
                # If parsing fails, wrap in basic structure
                result = {
                    "evaluated_issues": [{"description": result}]
                }
        
        return result
    
    def _format_results(
        self, 
        evaluated_result: Dict[str, Any], 
        document_info: Optional[Dict[str, Any]], 
        user_preferences: Dict[str, Any]
    ) -> str:
        """
        Format evaluated results into a report.
        
        Args:
            evaluated_result: Evaluated results
            document_info: Document metadata
            user_preferences: User preferences
            
        Returns:
            Formatted report (HTML string)
        """
        # Build context for formatting
        context = {
            "evaluated_result": evaluated_result,
            "document_info": document_info or {},
            "user_preferences": user_preferences or {}
        }
        
        # Execute formatting
        result = self.formatter_agent.execute_task(context=context)
        
        # Ensure result is a string (HTML)
        if not isinstance(result, str):
            if isinstance(result, dict):
                # Try to find HTML content in the result
                for key, value in result.items():
                    if isinstance(value, str) and value.startswith("<"):
                        return value
                
                # If no HTML found, convert to JSON string
                return json.dumps(result, indent=2)
            else:
                # Convert to string
                return str(result)
        
        return result
    
    def _review_results(
        self, 
        formatted_result: str, 
        document_info: Optional[Dict[str, Any]], 
        user_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Review formatted results.
        
        Args:
            formatted_result: Formatted results
            document_info: Document metadata
            user_preferences: User preferences
            
        Returns:
            Review results
        """
        # Build context for review
        context = {
            "formatted_result": formatted_result,
            "document_info": document_info or {},
            "user_preferences": user_preferences or {}
        }
        
        # Execute review
        result = self.reviewer_agent.execute_task(context=context)
        
        # Parse result if it's a string
        if isinstance(result, str):
            try:
                result = self._parse_json_safely(result)
            except:
                # If parsing fails, create a basic review structure
                result = {
                    "meets_requirements": None,
                    "summary": result,
                    "assessment": {}
                }
        
        return result
    
    def _parse_json_safely(self, text: str) -> Dict[str, Any]:
        """
        Parse JSON with fallbacks for common formats.
        
        Args:
            text: Text to parse
            
        Returns:
            Parsed JSON object
        """
        import re
        
        # Try direct parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        match = re.search(json_pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
        
        # Try to extract any JSON-like structure
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except:
            pass
        
        # Could not parse JSON, raise exception
        raise ValueError("Failed to parse JSON from response")
    
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