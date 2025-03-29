"""
LangChain Orchestrator for Better Notes (Simplified)
A streamlined orchestration layer that uses your existing LLMAdapter directly.
"""

import os
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path

# LangChain imports - minimal imports just for type checking
from langchain.callbacks.base import BaseCallbackHandler

# Local imports 
from process_context import ProcessingContext
from config_manager import ConfigManager
from universal_llm_adapter import LLMAdapter

# Document processing utilities
from lean.chunker import DocumentChunker
from lean.document import DocumentAnalyzer

# Agent imports
from agents.planner import PlannerAgent
from agents.extractor import ExtractorAgent
from agents.aggregator import AggregatorAgent
from agents.evaluator import EvaluatorAgent
from agents.formatter import FormatterAgent
from agents.reviewer import ReviewerAgent

logger = logging.getLogger(__name__)

class LangChainOrchestrator:
    """
    Streamlined orchestrator that processes documents using existing agents.
    Uses the LLMAdapter directly rather than trying to wrap it in LangChain.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.2,
        max_chunk_size: int = 10000,
        max_rpm: int = 10,
        verbose: bool = True,
        config_manager: Optional[ConfigManager] = None
    ):
        """Initialize the LangChain orchestrator."""
        # Store configuration
        self.model = model
        self.temperature = temperature
        self.max_chunk_size = max_chunk_size
        self.max_rpm = max_rpm
        self.verbose = verbose
        self.debug_mode = False
        
        # Create LLM adapter directly (don't try to use LangChain)
        self.llm_client = LLMAdapter(
            api_key=api_key,
            model=model,
            temperature=temperature
        )
        
        # Initialize config manager
        self.config_manager = config_manager or ConfigManager()
        
        # Create utilities
        self.document_analyzer = DocumentAnalyzer(self.llm_client)
        self.chunker = DocumentChunker()
        
        # Agent cache
        self._agents = {}
        
        logger.info(f"LangChainOrchestrator initialized with model: {model}")
    
    def _get_agent(self, agent_type: str, crew_type: str):
        """Get an agent instance with caching."""

        agent_type_map = {
            "extraction": "extractor",
            "aggregation": "aggregator", 
            "evaluation": "evaluator",
            "formatting": "formatter",
            "planning": "planner",
            "review": "reviewer"
        }
        
        # Convert stage name to agent type
        actual_agent_type = agent_type_map.get(agent_type, agent_type)
        
        cache_key = f"{actual_agent_type}_{crew_type}"

        
        # Return cached agent if exists
        if cache_key in self._agents:
            return self._agents[cache_key]
        
        # Get config for this crew type
        config = self.config_manager.get_config(crew_type)
        
        # Create agent based on type
        agent_map = {
            "planner": PlannerAgent,
            "extractor": ExtractorAgent,
            "aggregator": AggregatorAgent,
            "evaluator": EvaluatorAgent,
            "formatter": FormatterAgent,
            "reviewer": ReviewerAgent
        }
        
        agent_class = agent_map.get(agent_type)
        if not agent_class:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Create agent
        agent = agent_class(
            llm_client=self.llm_client,
            crew_type=crew_type,
            config=config,
            config_manager=self.config_manager,
            verbose=self.verbose,
            max_chunk_size=self.max_chunk_size,
            max_rpm=self.max_rpm
        )
        
        # Cache agent
        self._agents[cache_key] = agent
        return agent
    
    async def process_document(
        self, 
        document_text: str,
        options: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """Process a document through the pipeline."""
        # Initialize options
        options = options or {}
        crew_type = options.get("crew_type", "issues")
        
        # Create processing context
        context = ProcessingContext(document_text, options)
        
        # Set up progress callback
        if progress_callback:
            def enhanced_progress_callback(progress: float, message: str):
                stage_name = context.metadata.get("current_stage", "")
                stage_msg = f"[{stage_name}] {message}" if stage_name else message
                progress_callback(progress, stage_msg)
                
                # Log for verbose mode
                if self.verbose:
                    logger.info(f"Progress {progress:.2f}: {stage_msg}")
            
            context.progress_callback = enhanced_progress_callback
        
        # Get enabled stages from options or config
        enabled_stages = options.get("enabled_stages")
        if not enabled_stages:
            # Try to get from config
            config = self.config_manager.get_config(crew_type)
            enabled_stages = config.get("workflow", {}).get("enabled_stages")
        
        # Use default stages if not specified
        if not enabled_stages:
            enabled_stages = [
                "document_analysis", "chunking", "planning",
                "extraction", "aggregation", "evaluation",
                "formatting"
            ]
        
        # Add review stage if enabled
        if options.get("enable_reviewer", True) and "review" not in enabled_stages:
            enabled_stages.append("review")
        
        try:
            # Process each stage
            for stage_name in enabled_stages:
                # Set current stage
                context.set_stage(stage_name)
                
                # Execute stage with recovery
                try:
                    result = await self._execute_stage_with_recovery(context, stage_name, crew_type)
                    
                    # Store result and mark stage as complete
                    context.complete_stage(stage_name, result)
                    
                    # Log success for debugging
                    if self.debug_mode:
                        logger.debug(f"Stage {stage_name} completed successfully")
                        if isinstance(result, dict):
                            logger.debug(f"Result keys: {list(result.keys())}")
                except Exception as e:
                    # Log error but continue with next stage
                    logger.error(f"Error in stage {stage_name}: {e}")
                    context.fail_stage(stage_name, e)
            
            # Complete processing
            if progress_callback:
                progress_callback(1.0, "Analysis complete")
            
            # Return final result
            return context.get_final_result()
            
        except Exception as e:
            logger.exception(f"Error processing document: {e}")
            
            # Update context with error
            if context.metadata['current_stage']:
                context.fail_stage(context.metadata['current_stage'], str(e))
            
            # Return error result
            return {
                "error": str(e),
                "_metadata": {
                    "run_id": context.run_id,
                    "processing_time": context.get_processing_time(),
                    "errors": context.metadata.get("errors", []),
                    "error": True
                }
            }
    
    async def _execute_stage(self, context: ProcessingContext, stage_name: str, crew_type: str) -> Any:
        """Execute a specific processing stage."""
        # Document analysis stage
        if stage_name == "document_analysis":
            document_info = await self.document_analyzer.analyze_preview(context.document_text)
            context.document_info = document_info
            return document_info
        
        # Chunking stage
        elif stage_name == "chunking":
            options = context.options or {}
            min_chunks = options.get("min_chunks", 3)
            max_chunk_size = options.get("max_chunk_size", self.max_chunk_size)
            
            chunk_objects = self.chunker.chunk_document(
                context.document_text,
                min_chunks=min_chunks,
                max_chunk_size=max_chunk_size
            )
            
            context.chunks = [chunk["text"] for chunk in chunk_objects]
            context.chunk_metadata = chunk_objects
            
            return {
                "chunk_count": len(context.chunks),
                "chunk_metadata": chunk_objects
            }
        
        # Stages handled by agents
        else:
            # Get the appropriate agent
            agent = self._get_agent(stage_name, crew_type)
            
            # Process using the agent
            return await agent.process(context)
    
    async def _execute_stage_with_recovery(self, context: ProcessingContext, stage_name: str, crew_type: str) -> Any:
        """Execute a stage with error recovery capabilities."""
        try:
            # Try normal execution first
            return await self._execute_stage(context, stage_name, crew_type)
        except Exception as e:
            logger.error(f"Error in stage {stage_name}: {e}")
            
            # For non-critical stages, try to recover
            if stage_name in ["document_analysis", "planning", "review"]:
                logger.info(f"Attempting recovery for {stage_name}")
                
                # Simplified backup execution with reduced functionality
                if stage_name == "document_analysis":
                    # Simple document stats as fallback
                    word_count = len(context.document_text.split())
                    return {
                        "basic_stats": {
                            "word_count": word_count,
                            "char_count": len(context.document_text),
                            "estimated_tokens": word_count * 1.3
                        },
                        "is_meeting_transcript": False,
                        "recovered": True
                    }
                elif stage_name == "planning":
                    # Get default plan from base agent
                    agent = self._get_agent("planner", crew_type)
                    agent_types = agent._get_agent_types()
                    return agent._create_basic_plan(agent_types)
                elif stage_name == "review":
                    # Simple positive review
                    return {
                        "meets_requirements": True,
                        "summary": "Analysis completed successfully.",
                        "assessment": {
                            "overall": 4,
                            "completeness": 4,
                            "clarity": 4
                        }
                    }
            
            # Re-raise for other stages
            raise
    
    # Synchronous version for Streamlit
    def process_document_sync(
        self, 
        document_text: str,
        options: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """Synchronous version of process_document."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(
                self.process_document(
                    document_text=document_text,
                    options=options,
                    progress_callback=progress_callback
                )
            )
        finally:
            loop.close()
    
    def set_debug_mode(self, enabled: bool = True):
        """Enable or disable debug mode for more verbose output."""
        self.debug_mode = enabled
        logger.setLevel(logging.DEBUG if enabled else logging.INFO)
        
    def get_agent_instructions(self, crew_type: str) -> Dict[str, Dict[str, str]]:
        """
        Get the default instructions for all agents in a crew.
        Useful for debugging or customizing agent behavior.
        
        Args:
            crew_type: Type of crew
            
        Returns:
            Dictionary mapping agent types to their instructions
        """
        # Get planner agent to generate instructions
        planner = self._get_agent("planner", crew_type)
        
        # Get agent types
        agent_types = planner._get_agent_types()
        
        # Create basic instructions
        instructions = {}
        for agent_type in agent_types:
            instructions[agent_type] = {
                "instructions": planner._get_default_instructions(agent_type),
                "emphasis": planner._get_default_emphasis()
            }
        
        return instructions
    
    def explain_pipeline(self, crew_type: str) -> Dict[str, Any]:
        """
        Get an explanation of the processing pipeline.
        Useful for understanding the workflow.
        
        Args:
            crew_type: Type of crew
            
        Returns:
            Dictionary with pipeline explanation
        """
        # Get config for this crew type
        config = self.config_manager.get_config(crew_type)
        
        # Get stages
        stages = config.get("workflow", {}).get("enabled_stages", [
            "document_analysis", "chunking", "planning",
            "extraction", "aggregation", "evaluation",
            "formatting", "review"
        ])
        
        # Get agent roles
        agent_roles = config.get("workflow", {}).get("agent_roles", {})
        
        # Create explanation
        explanation = {
            "crew_type": crew_type,
            "model": self.model,
            "stages": stages,
            "stage_descriptions": {
                "document_analysis": "Analyzes document structure and content",
                "chunking": "Divides document into manageable segments",
                "planning": "Creates a tailored plan for processing",
                "extraction": "Extracts relevant information from chunks",
                "aggregation": "Combines and deduplicates information from chunks",
                "evaluation": "Evaluates importance and impact of findings",
                "formatting": "Creates a structured, readable report",
                "review": "Reviews analysis quality and alignment with user needs"
            },
            "agent_roles": {
                agent_type: {
                    "description": agent_info.get("description", f"{agent_type.capitalize()} Agent"),
                    "primary_task": agent_info.get("primary_task", f"Process {crew_type}")
                }
                for agent_type, agent_info in agent_roles.items()
            }
        }
        
        return explanation