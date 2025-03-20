# passes/passes.py
import json
import logging
import os
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from lean.async_openai_adapter import AsyncOpenAIAdapter  # Import

logger = logging.getLogger(__name__)

class PassProcessor:
    """Base class for all pass processors."""

    def __init__(self, llm_client, options=None):
        """
        Initialize the pass processor.

        Args:
            llm_client: LLM client for text analysis
            options: Processing options
        """
        self.llm_client = llm_client
        self.options = options

    async def process_document(self,
                               document_text: str,
                               document_info: Optional[Dict[str, Any]] = None,
                               progress_callback: Optional[Callable] = None,
                               prior_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a document with this pass.

        Args:
            document_text: Original document text
            document_info: Document metadata and context
            progress_callback: Optional callback for progress updates
            prior_result: Results from previous processing steps

        Returns:
            Dictionary with pass results
        """
        raise NotImplementedError("Subclasses must implement this method")


class TemplatedPass(PassProcessor):
    """Pass processor that uses JSON configuration templates."""

    def __init__(self, config_path: str, llm_client: AsyncOpenAIAdapter, options=None):
        """
        Initialize templated pass processor.

        Args:
            config_path: Path to JSON configuration file
            llm_client: LLM client for text analysis
            options: Processing options
        """
        super().__init__(llm_client, options)
        self.config_path = config_path
        self.config = self._load_config()  # Load config during initialization

    def _load_config(self) -> Dict[str, Any]:
        """Loads the JSON configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in configuration file: {self.config_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading pass configuration from {self.config_path}: {e}")
            raise

        # Validate configuration
        required_keys = ['pass_type', 'purpose', 'instructions', 'prompt_template']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required key '{key}' in pass configuration")
        return config

    async def process_document(self,
                               document_text: str,
                               document_info: Optional[Dict[str, Any]] = None,
                               progress_callback: Optional[Callable] = None,
                               prior_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a document using the configured template.

        Args:
            document_text: Original document text
            document_info: Document metadata and context
            progress_callback: Optional callback for progress updates
            prior_result: Results from previous processing steps (includes chunk summaries, etc.)

        Returns:
            Dictionary with pass results
        """
        # Prepare data for template (handle potential missing keys gracefully)
        template_data = {
            'document_text': document_text,
            'document_info': document_info or {},
            'summary': prior_result.get('synthesis_result', {}).get('summary', '') if prior_result else '',
            'chunk_summaries': prior_result.get('chunk_summaries', []) if prior_result else [],
            'options': self.options or {}  # Ensure options are available
        }

        # Fill in template with data
        try:
            prompt = self.config['prompt_template'].format(**template_data)
        except KeyError as e:
            logger.error(f"Error formatting prompt template: Missing key {e}")
            return {"result": {}, "error": f"Missing key in template: {e}"}  # Return error info

        # Process with structured output if schema is provided
        if 'output_schema' in self.config:
            try:
                result = await self.llm_client.generate_completion_with_structured_output(
                    prompt, self.config['output_schema']
                )
                if result is None:
                    logger.warning(f"Falling back to unstructured output for pass: {self.config['pass_type']}")
                    response = await self.llm_client.generate_completion_async(prompt)
                    result = {'raw_output': response}
            except Exception as e:
                logger.error(f"Error in structured output generation for pass {self.config['pass_type']}: {e}")
                return {"result": {}, "error": str(e)}  # Return error info
        else:
            # Regular text completion
            response = await self.llm_client.generate_completion_async(prompt)
            result = {'raw_output': response}

        # Add pass metadata
        result['pass_type'] = self.config['pass_type']
        result['pass_purpose'] = self.config['purpose']

        return {'result': result} # Consistent return format


def create_pass_processor(pass_type: str, llm_client: AsyncOpenAIAdapter, options=None) -> Optional[PassProcessor]:
    """
    Factory function to create a pass processor.

    Args:
        pass_type: Type of pass to create
        llm_client: LLM client for text analysis
        options: Processing options

    Returns:
        Pass processor instance or None if not found
    """
    config_dir = Path("passes/configurations")
    config_path = config_dir / f"{pass_type}.json"

    if config_path.exists():
        return TemplatedPass(str(config_path), llm_client, options)
    else:
        logger.error(f"No configuration found for pass type: {pass_type}")
        return None