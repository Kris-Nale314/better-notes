"""
Enhanced Issues Crew - Specialized crew for identifying issues in documents.
Uses the Planner Agent for optimized instructions and parallel processing for efficiency.
"""

from typing import Dict, Any, List, Optional, Union, Callable
import asyncio
import json
import logging
import os
import time
import traceback
from datetime import datetime

# Import agents
from agents.planner import PlannerAgent
from agents.extractor import ExtractorAgent
from agents.aggregator import AggregatorAgent
from agents.evaluator import EvaluatorAgent
from agents.formatter import FormatterAgent
from agents.reviewer import ReviewerAgent

logger = logging.getLogger(__name__)

class IssuesCrew:
   """
   Specialized crew for identifying, evaluating, and reporting issues in documents.
   """
   
   def __init__(
       self, 
       llm_client, 
       config_path=None, 
       verbose=True, 
       max_chunk_size=1500,
       max_rpm=10
   ):
       """Initialize the Issues Identification crew."""
       self.llm_client = llm_client
       self.verbose = verbose
       self.max_chunk_size = max_chunk_size
       self.max_rpm = max_rpm
       
       # Load configuration
       self.config = self._load_config(config_path)
       
       # Track process state
       self.process_state = {"status": "initialized", "stages": {}, "errors": []}
       self.start_time = None
       self.run_id = None
       
       # Create the planner agent
       self.planner = PlannerAgent(
           llm_client=llm_client,
           config=self.config,
           verbose=verbose,
           max_chunk_size=max_chunk_size,
           max_rpm=max_rpm
       )
       
       # Initialize specialized agents
       self.extractor_agent = ExtractorAgent(
           llm_client=llm_client,
           crew_type="issues",
           config=self.config,
           verbose=verbose,
           max_chunk_size=max_chunk_size,
           max_rpm=max_rpm
       )
       
       self.aggregator_agent = AggregatorAgent(
           llm_client=llm_client,
           crew_type="issues", 
           config=self.config,
           verbose=verbose,
           max_chunk_size=max_chunk_size,
           max_rpm=max_rpm
       )
       
       self.evaluator_agent = EvaluatorAgent(
           llm_client=llm_client,
           crew_type="issues",
           config=self.config,
           verbose=verbose,
           max_chunk_size=max_chunk_size,
           max_rpm=max_rpm
       )
       
       self.formatter_agent = FormatterAgent(
           llm_client=llm_client,
           crew_type="issues",
           config=self.config,
           verbose=verbose,
           max_chunk_size=max_chunk_size,
           max_rpm=max_rpm
       )
       
       self.reviewer_agent = ReviewerAgent(
           llm_client=llm_client,
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
       for agent_name in ["planner", "extractor_agent", "aggregator_agent", 
                         "evaluator_agent", "formatter_agent", "reviewer_agent"]:
           agent = getattr(self, agent_name)
           if hasattr(agent, "agent") and hasattr(agent.agent, "max_rpm"):
               agent.agent.max_rpm = new_rpm
   
   def _load_config(self, config_path=None):
       """Load the configuration file."""
       if not config_path:
           config_path = os.path.join(
               os.path.dirname(os.path.dirname(__file__)),
               "agents", "config", "issues_config.json"
           )
       
       try:
           with open(config_path, 'r') as f:
               config = json.load(f)
               logger.info(f"Loaded configuration from {config_path}")
               return config
       except (FileNotFoundError, json.JSONDecodeError) as e:
           logger.error(f"Error loading config: {str(e)}")
           return {}
   
   def process_document(
       self, 
       document_text_or_chunks,
       document_info: Optional[Dict[str, Any]] = None, 
       user_preferences: Optional[Dict[str, Any]] = None, 
       max_chunk_size: Optional[int] = None,
       enable_reviewer: bool = True,
       progress_callback: Optional[Callable] = None
   ) -> Dict[str, Any]:
       """Process a document to identify issues."""
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
               user_preferences
           )
           
           self.process_state["document_length"] = len(document_text_or_chunks) if isinstance(document_text_or_chunks, str) else sum(len(c) for c in document_text_or_chunks)
           self.process_state["chunk_count"] = len(chunks)
           self._complete_stage("document_preparation")
           
           # Stage 2: Create plan with Planner agent
           self._start_stage("planning")
           if progress_callback:
               progress_callback(0.1, "Creating analysis plan...")
           
           agent_plan = self.planner.create_plan(
               document_info=document_info or {},
               user_preferences=user_preferences or {},
               crew_type="issues"
           )
           
           # Apply plan to each agent
           self._apply_agent_plan(agent_plan)
           self._complete_stage("planning")
           
           # Stage 3: Extract issues from chunks in parallel
           self._start_stage("extraction")
           if progress_callback:
               progress_callback(0.2, "Extracting issues from document chunks...")
           
           extraction_results = self._extract_from_chunks(chunks, chunk_metadata, document_info, progress_callback)
           
           # Critical fix: Ensure each extraction result is properly validated
           validated_results = []
           for result in extraction_results:
               if isinstance(result, str):
                   try:
                       # Try to parse string as JSON
                       parsed = json.loads(result)
                       validated_results.append(parsed)
                   except json.JSONDecodeError:
                       # Create a simple dict with the string content
                       validated_results.append({"text": result})
               else:
                   validated_results.append(result)
           
           extraction_results = validated_results
           
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
               progress_callback(0.5, "Aggregating issues...")
           
           aggregated_result = self._aggregate_results(extraction_results, document_info)
           self._complete_stage("aggregation")
           
           # Stage 5: Evaluation
           self._start_stage("evaluation")
           if progress_callback:
               progress_callback(0.6, "Evaluating issues...")
           
           evaluated_result = self._evaluate_results(aggregated_result, document_info)
           self._complete_stage("evaluation")
           
           # Stage 6: Formatting
           self._start_stage("formatting")
           if progress_callback:
               progress_callback(0.7, "Formatting report...")
           
           formatted_result = self._format_results(evaluated_result, document_info, user_preferences)
           self._complete_stage("formatting")
           
           # Stage 7: Review (optional)
           final_result = formatted_result
           if enable_reviewer:
               self._start_stage("review")
               if progress_callback:
                   progress_callback(0.8, "Reviewing report...")
               
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
               progress_callback(1.0, f"Analysis complete in {duration:.2f} seconds")
           
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
       user_preferences
   ):
       """Prepare document by chunking or using provided chunks."""
       chunk_metadata = []
       
       if isinstance(document_text_or_chunks, str):
           # Get chunking parameters
           min_chunks = user_preferences.get("min_chunks", 3) if user_preferences else 3
           
           # We have full text - chunk it
           from lean.chunker import DocumentChunker
           chunker = DocumentChunker()
           
           # Calculate appropriate chunk size
           if self.max_chunk_size is None:
               doc_length = len(document_text_or_chunks)
               calculated_chunk_size = doc_length // min_chunks
               self.max_chunk_size = min(calculated_chunk_size + 100, 16000)
           
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
           
           # Analyze document if needed
           if not document_info:
               from lean.document import DocumentAnalyzer
               analyzer = DocumentAnalyzer(self.llm_client)
               document_info = asyncio.run(analyzer.analyze_preview(document_text_or_chunks))
               document_info["original_text_length"] = len(document_text_or_chunks)
       
       elif isinstance(document_text_or_chunks, list):
           # We already have chunks
           chunks = document_text_or_chunks
           chunk_metadata = [{"index": i} for i in range(len(chunks))]
       else:
           raise TypeError("Expected either a string or a list of chunks")
       
       return chunks, chunk_metadata
   
   def _apply_agent_plan(self, agent_plan):
       """Apply plan from Planner to each specialized agent."""
       for agent_type, instructions in agent_plan.items():
           # Skip metadata and non-agent entries
           if agent_type.startswith('_') or agent_type not in [
               "extraction", "aggregation", "evaluation", "formatting", "reviewer"
           ]:
               continue
           
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
                   logger.info(f"Applied plan instructions to {agent_type} agent")
   
   def _extract_from_chunks(self, chunks, chunk_metadata, document_info, progress_callback):
       """Extract issues from chunks in parallel with improved error handling."""
       async def extract_all_chunks():
           # Set up concurrency control
           max_concurrency = min(len(chunks), 10)
           semaphore = asyncio.Semaphore(max_concurrency)
           
           async def process_chunk(idx):
               async with semaphore:
                   chunk = chunks[idx]
                   metadata = chunk_metadata[idx] if idx < len(chunk_metadata) else {}
                   
                   # Log progress
                   if self.verbose:
                       logger.info(f"Processing chunk {idx+1}/{len(chunks)}")
                   
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
           
           # Update progress as chunks complete
           completed = 0
           results = []
           for future in asyncio.as_completed(tasks):
               result = await future
               results.append(result)
               
               completed += 1
               if progress_callback:
                   # Scale progress from 0.2 to 0.5 during extraction
                   extraction_progress = 0.2 + (0.3 * (completed / len(chunks)))
                   progress_callback(extraction_progress, f"Extracted {completed}/{len(chunks)} chunks")
           
           # Sort results by chunk index if metadata is available
           return sorted(results, key=lambda r: r.get("_metadata", {}).get("chunk_index", 0) if isinstance(r, dict) else 0)
       
       # Run the extraction process
       try:
           return asyncio.run(extract_all_chunks())
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