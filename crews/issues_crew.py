# crews/issues_crew.py - updated to work with orchestrator LLM client
from typing import Dict, Any, List, Optional
from crewai import Crew, Task
import os
import json
import logging

from agents.extractor import ExtractorAgent
from agents.aggregator import AggregatorAgent
from agents.evaluator import EvaluatorAgent
from agents.formatter import FormatterAgent

logger = logging.getLogger(__name__)

class IssuesCrew:
    """
    A specialized crew for identifying, evaluating, and reporting issues in documents.
    Coordinates multiple agents working together to produce comprehensive issue analysis.
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
            llm_client: LLM client from orchestrator
            config_path: Optional path to custom configuration
            verbose: Whether to enable verbose mode for agents and crew
            max_chunk_size: Maximum size of text chunks to process
            max_rpm: Maximum requests per minute for API rate limiting
        """
        self.llm_client = llm_client
        self.verbose = verbose
        self.max_chunk_size = max_chunk_size
        self.max_rpm = max_rpm
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Create the agents
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
        
        # Create the crew
        self.crew = Crew(
            agents=[
                self.extractor_agent.agent,
                self.aggregator_agent.agent,
                self.evaluator_agent.agent,
                self.formatter_agent.agent
            ],
            tasks=[],  # Will be created for each document
            verbose=verbose
        )
    
    # Add method to update RPM settings
    def update_rpm(self, new_rpm: int) -> None:
        """
        Update the maximum requests per minute for all agents.
        
        Args:
            new_rpm: New maximum requests per minute
        """
        self.max_rpm = new_rpm
        
        # Update all agents
        for agent_type in ["extractor", "aggregator", "evaluator", "formatter"]:
            agent = getattr(self, f"{agent_type}_agent")
            if hasattr(agent, "agent") and hasattr(agent.agent, "max_rpm"):
                agent.agent.max_rpm = new_rpm
    
    
    def _load_config(self, config_path=None):
        """
        Load the configuration file.
        
        Args:
            config_path: Optional custom path to config file
            
        Returns:
            Configuration dictionary
        """
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
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            return {}
    
    def process_document(self, document_chunks, document_info=None, user_preferences=None, max_chunk_size=None):
        """
        Process a document to identify issues.
        
        Args:
            document_chunks: List of document chunks to analyze
            document_info: Optional document metadata
            user_preferences: Optional user preferences for analysis
            max_chunk_size: Optional override for maximum chunk size
            
        Returns:
            Processed result with identified, evaluated, and formatted issues
        """
        # Update max chunk size if provided
        if max_chunk_size is not None:
            self.max_chunk_size = max_chunk_size
            # Update agents with new max chunk size
            for agent in [self.extractor_agent, self.aggregator_agent, self.evaluator_agent, self.formatter_agent]:
                agent.max_chunk_size = max_chunk_size
        
        logger.info(f"Starting issues analysis with {len(document_chunks)} chunks")
        
        # Create extraction tasks for each chunk
        extraction_tasks = self._create_extraction_tasks(document_chunks, document_info)
        
        # Create aggregation task
        aggregation_task = self._create_aggregation_task(document_info)
        
        # Create evaluation task
        evaluation_task = self._create_evaluation_task(document_info)
        
        # Create formatting task
        formatting_task = self._create_formatting_task(document_info, user_preferences)
        
        # Update crew tasks
        self.crew.tasks = extraction_tasks + [aggregation_task, evaluation_task, formatting_task]
        
        # Execute the crew's work
        logger.info("Executing issues crew workflow")
        result = self.crew.kickoff()
        
        logger.info("Issues analysis complete")
        return result
    
    def _create_extraction_tasks(self, document_chunks, document_info):
        """
        Create extraction tasks for each document chunk.
        
        Args:
            document_chunks: List of document chunks
            document_info: Document metadata
            
        Returns:
            List of extraction tasks
        """
        tasks = []
        for i, chunk in enumerate(document_chunks):
            # Ensure the chunk isn't too large
            safe_chunk = self.extractor_agent.truncate_text(chunk, self.max_chunk_size)
            
            task = Task(
                description=f"Analyze document chunk {i+1}/{len(document_chunks)} to identify issues:\n\n{safe_chunk}",
                agent=self.extractor_agent.agent,
                expected_output="A list of issues found in the document chunk.",
                async_execution=False,
                output_file=f"extraction_result_{i}.json"
            )
            tasks.append(task)
        
        return tasks
    
    def _create_aggregation_task(self, document_info):
        """
        Create the aggregation task.
        
        Args:
            document_info: Document metadata
            
        Returns:
            Aggregation task
        """
        # Create a compact version of document info to prevent large headers
        safe_doc_info = ""
        if document_info:
            try:
                safe_doc_info = json.dumps(document_info, indent=None)[:500]
            except:
                safe_doc_info = str(document_info)[:500]
        
        return Task(
            description=f"Combine all identified issues from the extraction agents. "
                      f"Remove duplicates but track frequency of mentions. "
                      f"Create a consolidated list of unique issues.\n\n"
                      f"Document context: {safe_doc_info}",
            agent=self.aggregator_agent.agent,
            expected_output="A consolidated list of unique issues with frequency counts.",
            async_execution=False,
            output_file="aggregated_issues.json"
        )
    
    def _create_evaluation_task(self, document_info):
        """
        Create the evaluation task.
        
        Args:
            document_info: Document metadata
            
        Returns:
            Evaluation task
        """
        # Extract criteria from config for prompt, but keep it compact
        criteria = self.config.get("evaluation", {}).get("criteria", {})
        criteria_text = ""
        for level, description in list(criteria.items())[:2]:  # Limit to just a couple of examples
            criteria_text += f"{level.upper()}: {description[:100]}...\n"
        
        return Task(
            description=f"Evaluate each issue and assign a severity level (critical, high, medium, or low).\n\n"
                      f"Use these criteria as examples:\n{criteria_text}\n\n"
                      f"Use the output from the aggregation task as your input.",
            agent=self.evaluator_agent.agent,
            expected_output="The list of issues with severity levels assigned to each.",
            async_execution=False,
            output_file="evaluated_issues.json"
        )
    
    def _create_formatting_task(self, document_info, user_preferences):
        """
        Create the formatting task.
        
        Args:
            document_info: Document metadata
            user_preferences: User formatting preferences
            
        Returns:
            Formatting task
        """
        # Create compact versions to prevent large headers
        safe_preferences = ""
        if user_preferences:
            try:
                safe_preferences = json.dumps(user_preferences, indent=None)[:300]
            except:
                safe_preferences = str(user_preferences)[:300]
        
        format_template_preview = self.formatter_agent.get_format_template()[:500] + "..."
        
        return Task(
            description=f"Format the evaluated issues into a structured report.\n\n"
                      f"USER PREFERENCES: {safe_preferences}\n\n"
                      f"Follow this format template: {format_template_preview}\n\n"
                      f"Use the output from the evaluation task as your input.",
            agent=self.formatter_agent.agent,
            expected_output="A fully formatted issues report.",
            async_execution=False,
            output_file="final_issues_report.md"
        )