# crews/issues_crew.py
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
    
    def __init__(self, llm_client, config_path=None, verbose=True):
        """
        Initialize the Issues Identification crew.
        
        Args:
            llm_client: LLM client for agent communication
            config_path: Optional path to custom configuration
            verbose: Whether to enable verbose mode for agents and crew
        """
        self.llm_client = llm_client
        self.verbose = verbose
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Create the agents
        self.extractor_agent = ExtractorAgent(
            llm_client=llm_client,
            crew_type="issues",
            config=self.config,
            verbose=verbose
        )
        
        self.aggregator_agent = AggregatorAgent(
            llm_client=llm_client,
            crew_type="issues", 
            config=self.config,
            verbose=verbose
        )
        
        self.evaluator_agent = EvaluatorAgent(
            llm_client=llm_client,
            crew_type="issues",
            config=self.config,
            verbose=verbose
        )
        
        self.formatter_agent = FormatterAgent(
            llm_client=llm_client,
            crew_type="issues",
            config=self.config,
            verbose=verbose
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
    
    def process_document(self, document_chunks, document_info=None, user_preferences=None):
        """
        Process a document to identify issues.
        
        Args:
            document_chunks: List of document chunks to analyze
            document_info: Optional document metadata
            user_preferences: Optional user preferences for analysis
            
        Returns:
            Processed result with identified, evaluated, and formatted issues
        """
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
            task = Task(
                description=f"Analyze document chunk {i+1}/{len(document_chunks)} to identify all potential issues:\n\n"
                          f"CHUNK TEXT:\n{chunk}",
                agent=self.extractor_agent.agent,
                expected_output="A list of all potential issues found in the document chunk, "
                               "with descriptions and context.",
                output_file=f"extraction_result_{i}.json"  # Save intermediate results
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
        return Task(
            description="Combine all identified issues from the extraction agents. "
                      "Remove duplicates but track frequency of mentions. "
                      "Create a consolidated list of unique issues that preserves context. "
                      "Use the outputs from all extraction tasks.",
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
        # Extract criteria from config for prompt
        criteria = self.config.get("evaluation", {}).get("criteria", {})
        criteria_text = ""
        for level, description in criteria.items():
            criteria_text += f"{level.upper()} issues are those that: {description}\n\n"
        
        return Task(
            description=f"Evaluate each issue and assign a severity level (critical, high, medium, or low).\n\n"
                      f"Use these criteria for evaluation:\n{criteria_text}\n\n"
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
        # Get format template from config
        format_template = self.config.get("formatting", {}).get("format_template", "")
        
        preferences_text = ""
        if user_preferences:
            preferences_text = f"\n\nUSER PREFERENCES:\n{json.dumps(user_preferences, indent=2)}"
        
        return Task(
            description=f"Format the evaluated issues into a structured report using this template:\n\n"
                      f"{format_template}\n\n"
                      f"DOCUMENT INFORMATION:\n{json.dumps(document_info or {}, indent=2)}"
                      f"{preferences_text}\n\n"
                      f"Use the output from the evaluation task as your input.",
            agent=self.formatter_agent.agent,
            expected_output="A fully formatted issues report according to the specified template.",
            async_execution=False,
            output_file="final_issues_report.md"
        )