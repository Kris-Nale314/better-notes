"""
Streamlined LangChain pipeline for Better Notes.
A modular pipeline that leverages LangChain with your existing agent architecture.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import time

# LangChain imports
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

# Local imports
from universal_llm_adapter import LLMAdapter
from config_manager import ConfigManager
from process_context import ProcessingContext

# Agent imports (using existing agents)
from agents.base import BaseAgent
from agents.planner import PlannerAgent
from agents.extractor import ExtractorAgent
from agents.aggregator import AggregatorAgent
from agents.evaluator import EvaluatorAgent
from agents.formatter import FormatterAgent
from agents.reviewer import ReviewerAgent

# Import utilities
from lean.chunker import DocumentChunker
from lean.document import DocumentAnalyzer

logger = logging.getLogger(__name__)

class LangChainAdapter:
    """Adapter to use our existing agents with LangChain."""
    
    def __init__(self, llm_chain):
        self.llm_chain = llm_chain
    
    async def generate_completion_async(self, prompt: str) -> str:
        """Generate a completion using LangChain."""
        result = self.llm_chain.run(prompt=prompt)
        return result
    
    async def generate_structured_output(self, prompt: str, output_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured output using LangChain."""
        result = self.llm_chain.run(prompt=prompt)
        
        # Parse result to match expected schema
        adapter = LLMAdapter(None)  # We just need its parsing methods
        return adapter.parse_json(result, output_schema)

class StreamlinedLangChainPipeline:
    """
    A streamlined LangChain pipeline that reuses existing agents.
    Provides a cleaner architecture while maintaining compatibility.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.2,
        max_chunk_size: int = 10000,
        max_rpm: int = 10,
        verbose: bool = True,
        config_manager = None
    ):
        """Initialize the streamlined pipeline."""
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            api_key=api_key,
            model_name=model,
            temperature=temperature
        )
        
        # Create base chain for LangChain adapter
        system_message = SystemMessagePromptTemplate.from_template("You are a helpful AI assistant.")
        human_message = HumanMessagePromptTemplate.from_template("{prompt}")
        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        self.base_chain = LLMChain(llm=self.llm, prompt=chat_prompt, output_key="output")
        
        # Create LangChain adapter for our agents
        self.langchain_adapter = LangChainAdapter(self.base_chain)
        
        # Store configuration
        self.model = model
        self.temperature = temperature
        self.max_chunk_size = max_chunk_size
        self.max_rpm = max_rpm
        self.verbose = verbose
        
        # Initialize config manager
        self.config_manager = config_manager or ConfigManager.get_instance()
        
        # Create document utilities
        self.document_analyzer = DocumentAnalyzer(self.langchain_adapter)
        self.chunker = DocumentChunker()
        
        # Cache for created agents
        self._agents = {}
        
        logger.info(f"Initialized StreamlinedLangChainPipeline with model: {model}")
    
    def create_agent(self, agent_type: str, crew_type: str) -> BaseAgent:
        """
        Create an agent of the specified type.
        Uses caching to avoid recreating agents.
        
        Args:
            agent_type: Type of agent to create
            crew_type: Type of crew the agent belongs to
            
        Returns:
            The created agent
        """
        # Create cache key
        cache_key = f"{agent_type}_{crew_type}"
        
        # Check cache
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
            llm_client=self.langchain_adapter,
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
        """
        Process a document through the LangChain pipeline.
        
        Args:
            document_text: Document text to process
            options: Processing options
            progress_callback: Progress callback
            
        Returns:
            Processing results
        """
        # Resolve options with defaults
        options = options or {}
        crew_type = options.get("crew_type", "issues")
        
        # Create processing context
        context = ProcessingContext(document_text, options)
        
        # Create progress tracker
        stages = [
            ("document_analysis", 0.05, "Analyzing document..."),
            ("chunking", 0.12, "Chunking document..."),
            ("planning", 0.17, "Creating analysis plan..."),
            ("extraction", 0.45, "Extracting from document..."),
            ("aggregation", 0.65, "Aggregating results..."),
            ("evaluation", 0.75, "Evaluating results..."),
            ("formatting", 0.85, "Formatting report..."),
            ("review", 0.95, "Reviewing analysis...")
        ]
        
        try:
            # Execute each stage
            for stage_name, progress_value, message in stages:
                # Skip review stage if disabled
                if stage_name == "review" and not options.get("enable_reviewer", True):
                    continue
                
                # Update progress
                if progress_callback:
                    progress_callback(progress_value, message)
                
                # Set current stage
                context.set_stage(stage_name)
                
                # Execute stage
                result = await self._execute_stage(context, stage_name, crew_type)
                
                # Store result and mark stage as complete
                context.complete_stage(stage_name, result)
            
            # Complete processing
            if progress_callback:
                progress_callback(1.0, "Analysis complete")
            
            # Return the final result
            return context.get_final_result()
            
        except Exception as e:
            logger.exception(f"Error processing document: {e}")
            
            # Update context with error
            if hasattr(context, 'metadata') and 'current_stage' in context.metadata:
                current_stage = context.metadata['current_stage']
                context.fail_stage(current_stage, str(e))
            
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
        """
        Execute a processing stage.
        
        Args:
            context: Processing context
            stage_name: Name of the stage to execute
            crew_type: Type of crew
            
        Returns:
            Stage result
        """
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
        
        # Agent stages
        else:
            # Map stage name to agent type if needed
            agent_type = stage_name
            
            # Create the agent
            agent = self.create_agent(agent_type, crew_type)
            
            # Execute the agent
            return await agent.process(context)
    
    # Synchronous version for easier integration
    def process_document_sync(
        self, 
        document_text: str,
        options: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Synchronous version of process_document.
        
        Args:
            document_text: Document text to process
            options: Processing options
            progress_callback: Progress callback
            
        Returns:
            Processing results
        """
        # Create and run event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run async method
            return loop.run_until_complete(
                self.process_document(
                    document_text=document_text,
                    options=options,
                    progress_callback=progress_callback
                )
            )
        finally:
            # Always close the loop
            loop.close()


# Example usage
if __name__ == "__main__":
    import os
    
    # Define a simple progress callback
    def print_progress(progress, message):
        print(f"[{progress:.1%}] {message}")
    
    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        exit(1)
    
    # Sample document
    sample_document = """
    Project Status Report: Database Migration Project
    
    Current Status: At Risk
    
    Executive Summary:
    The database migration project is currently behind schedule due to several technical challenges and resource constraints. The original timeline estimated completion by November 15, but current projections indicate a delay of at least 3 weeks.
    
    Key Issues:
    1. Technical Challenges: The legacy database structure has more inconsistencies than initially documented, requiring additional cleansing scripts.
    2. Resource Constraints: The DBA team is currently understaffed, with two key members being pulled into other critical projects.
    3. Integration Testing Failures: Initial testing revealed compatibility issues with three downstream systems that weren't identified in the planning phase.
    4. Budget Concerns: Additional licensing costs for the migration tools were not accounted for in the initial budget, creating a projected overage of $45,000.
    
    Mitigation Plans:
    - Requesting additional DBA resources from the enterprise pool
    - Developing workarounds for the integration issues with downstream systems
    - Evaluating alternative migration tools with lower licensing costs
    
    Next Steps:
    1. Meeting with steering committee to approve revised timeline
    2. Finalizing resource reallocation plan
    3. Completing comprehensive testing plan for downstream systems
    
    Please provide feedback on the proposed mitigation strategies by Friday.
    """
    
    # Process document
    pipeline = StreamlinedLangChainPipeline(api_key=api_key, model="gpt-4")
    result = pipeline.process_document_sync(
        sample_document,
        options={"crew_type": "issues", "detail_level": "standard"},
        progress_callback=print_progress
    )
    
    # Print result
    print("\nResult:", result)