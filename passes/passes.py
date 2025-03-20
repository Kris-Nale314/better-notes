# passes/passes.py
import json
import logging
import os
import asyncio
import re
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path

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

    def __init__(self, config_path: str, llm_client, options=None):
        """
        Initialize templated pass processor.

        Args:
            config_path: Path to JSON configuration file
            llm_client: LLM client for text analysis
            options: Processing options
        """
        super().__init__(llm_client, options)
        self.config_path = config_path
        self.config = self._load_config()

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
            if key not in config:
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
        # Ensure we have document_info
        if document_info is None:
            document_info = {}
            
        # Get pass-specific options if available
        pass_options = {}
        if self.options and hasattr(self.options, 'get_pass_options'):
            pass_options = self.options.get_pass_options(self.config['pass_type'])
            logger.info(f"Using pass options for {self.config['pass_type']}: {pass_options}")
            
        # Determine if we should process chunks or use the summary
        process_by_chunks = self.config.get('process_by_chunks', False)
        
        # Get chunk info from prior result
        chunk_summaries = []
        if prior_result and 'chunk_summaries' in prior_result:
            chunk_summaries = prior_result['chunk_summaries']
        
        # Get summary from prior result
        summary = ""
        if prior_result and 'synthesis_result' in prior_result:
            synthesis_result = prior_result['synthesis_result']
            summary = synthesis_result.get('summary', '')
        elif prior_result and 'summary' in prior_result:
            summary = prior_result['summary']
        
        try:
            if progress_callback:
                progress_callback(0.1, f"Starting {self.config['pass_type']} pass...")
                
            if process_by_chunks and chunk_summaries:
                # Process each chunk individually and combine results
                chunk_results = []
                total_chunks = len(chunk_summaries)
                
                for i, chunk in enumerate(chunk_summaries):
                    # Report progress if callback provided
                    if progress_callback:
                        progress = 0.1 + (0.7 * (i / total_chunks))
                        message = f"Processing chunk {i+1}/{total_chunks} for {self.config['pass_type']} pass"
                        progress_callback(progress, message)
                    
                    # Process this chunk
                    chunk_result = await self._process_chunk(chunk, document_info, prior_result)
                    chunk_results.append(chunk_result)
                
                # Combine results from all chunks
                if progress_callback:
                    progress_callback(0.8, f"Combining results for {self.config['pass_type']} pass")
                
                combined_result = await self._combine_chunk_results(chunk_results, document_info)
                return {'result': combined_result}
            else:
                # Process using the summary
                if progress_callback:
                    progress_callback(0.3, f"Analyzing document for {self.config['pass_type']} pass")
                
                # Create template data dictionary with all variables
                template_data = {
                    'document_text': document_text[:10000],  # Limit length to avoid token limits
                    'document_info': document_info,
                    'summary': summary,
                    'options': pass_options
                }
                
                # Format the template using safe formatter
                try:
                    prompt = self._format_template(self.config['prompt_template'], template_data)
                except Exception as e:
                    logger.error(f"Error formatting template: {e}")
                    return {'result': {'error': f"Template formatting error: {str(e)}"}}
                
                if progress_callback:
                    progress_callback(0.5, f"Processing {self.config['pass_type']} with LLM...")
                
                # Generate completion
                try:
                    response = await self.llm_client.generate_completion_async(prompt)
                    
                    # Try to extract structured data (JSON)
                    result = self._extract_json(response)
                    
                    # If no structured data found, use the raw response
                    if not result:
                        result = {'raw_output': response}
                    
                    # Add pass metadata
                    result['pass_type'] = self.config['pass_type']
                    
                    if progress_callback:
                        progress_callback(0.9, f"Completed {self.config['pass_type']} pass")
                        
                    return {'result': result}
                    
                except Exception as e:
                    logger.error(f"Error in LLM processing: {e}")
                    return {'result': {'error': str(e)}}
                
        except Exception as e:
            logger.error(f"Error in pass processing: {str(e)}")
            return {'result': {'error': str(e)}}

    def _format_template(self, template: str, data: Dict[str, Any]) -> str:
        """
        Format template with data, handling complex nested structures.
        
        Args:
            template: Template string
            data: Data dictionary
            
        Returns:
            Formatted string
        """
        # Simple string formatter that handles missing keys and nested dicts
        result = template
        
        # Replace {variable} placeholders
        pattern = r'\{([^{}]+)\}'
        matches = re.findall(pattern, template)
        
        for match in matches:
            try:
                # Handle nested dictionary access with dot notation (e.g., document_info.client_name)
                if '.' in match:
                    parts = match.split('.')
                    value = data
                    for part in parts:
                        if part in value:
                            value = value[part]
                        else:
                            value = ''
                            break
                # Handle dictionary get with default (e.g., options.get('key', 'default'))
                elif 'get(' in match:
                    # This is a simplistic approach - a real implementation would use eval or similar
                    dict_name, rest = match.split('.get(', 1)
                    key, default = rest.rstrip(')').split(',', 1)
                    key = key.strip().strip('"').strip("'")
                    default = default.strip()
                    
                    if dict_name in data and isinstance(data[dict_name], dict):
                        value = data[dict_name].get(key, default)
                    else:
                        value = default
                # Handle conditional expressions (e.g., document_info.get('is_transcript', False) and 'Yes' or 'No')
                elif ' and ' in match or ' or ' in match:
                    # This is very simplistic - just handle common patterns
                    if ' and ' in match and ' or ' in match:
                        condition, rest = match.split(' and ', 1)
                        true_value, false_value = rest.split(' or ', 1)
                        
                        # Evaluate condition
                        condition_value = False
                        if '.' in condition:
                            parts = condition.split('.')
                            value = data
                            for part in parts:
                                if part in value:
                                    value = value[part]
                                else:
                                    value = False
                                    break
                            condition_value = bool(value)
                        
                        value = true_value if condition_value else false_value
                        # Strip quotes
                        value = value.strip().strip('"').strip("'")
                    else:
                        # Default handling
                        value = ''
                else:
                    # Simple variable
                    value = data.get(match, '')
                    
                # Replace in template
                placeholder = '{' + match + '}'
                result = result.replace(placeholder, str(value))
                
            except Exception as e:
                logger.warning(f"Error formatting template variable {match}: {e}")
                # Replace with empty string
                placeholder = '{' + match + '}'
                result = result.replace(placeholder, '')
                
        return result

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON from text response.
        
        Args:
            text: Response text potentially containing JSON
            
        Returns:
            Extracted JSON data or empty dict
        """
        # Check for JSON in code blocks
        json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        match = re.search(json_pattern, text)
        if match:
            try:
                json_str = match.group(1)
                return json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON in code block")
                
        # Try to find JSON without code blocks
        try:
            # Look for objects surrounded by curly braces
            brace_pattern = r'\{[\s\S]*?\}'
            for match in re.finditer(brace_pattern, text):
                try:
                    json_str = match.group(0)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
                    
            # If no valid JSON found, return empty dict
            return {}
            
        except Exception as e:
            logger.warning(f"Error extracting JSON: {e}")
            return {}

    async def _process_chunk(self,
                           chunk: Dict[str, Any],
                           document_info: Dict[str, Any],
                           prior_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a single chunk.
        
        Args:
            chunk: The chunk to process
            document_info: Document metadata and context
            prior_result: Results from previous processing steps
            
        Returns:
            Dictionary with chunk processing results
        """
        # Extract chunk details
        chunk_text = chunk.get('text', '')
        chunk_summary = chunk.get('summary', '')
        chunk_position = chunk.get('position', 'unknown')
        chunk_index = chunk.get('chunk_index', 0)
        
        # Get the appropriate prompt template
        if 'chunk_prompt_template' in self.config:
            prompt_template = self.config['chunk_prompt_template']
        else:
            prompt_template = self.config['prompt_template']
        
        # Create template data
        template_data = {
            'chunk_text': chunk_text,
            'chunk_summary': chunk_summary,
            'chunk_position': chunk_position,
            'chunk_index': chunk_index,
            'document_info': document_info
        }
        
        # Format prompt
        try:
            prompt = self._format_template(prompt_template, template_data)
        except Exception as e:
            logger.error(f"Error formatting chunk prompt: {e}")
            return {"error": str(e), "chunk_index": chunk_index}
        
        # Generate completion
        try:
            response = await self.llm_client.generate_completion_async(prompt)
            
            # Try to extract structured data
            result = self._extract_json(response)
            
            # If no structured data found, use raw response
            if not result:
                result = {'raw_output': response}
                
            # Add chunk metadata
            result['chunk_index'] = chunk_index
            result['chunk_position'] = chunk_position
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_index}: {e}")
            return {"error": str(e), "chunk_index": chunk_index}

    async def _combine_chunk_results(self, 
                                  chunk_results: List[Dict[str, Any]], 
                                  document_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine results from multiple chunks.
        
        Args:
            chunk_results: Results from processing individual chunks
            document_info: Document metadata and context
            
        Returns:
            Combined result
        """
        pass_type = self.config['pass_type']
        
        # Choose the appropriate combination method based on pass type
        if pass_type == 'issue_identification':
            return await self._combine_issues(chunk_results, document_info)
        elif pass_type == 'opportunity_identification':
            return await self._combine_opportunities(chunk_results, document_info)
        elif pass_type == 'action_items':
            return await self._combine_actions(chunk_results, document_info)
        else:
            # Generic combination - just combine all chunks
            all_data = []
            for result in chunk_results:
                if 'error' in result:
                    # Skip chunks with errors
                    continue
                    
                # Add chunk data
                all_data.append(result)
                
            return {
                "chunks": all_data,
                "pass_type": pass_type,
                "summary": f"Processed {len(chunk_results)} chunks with {len(all_data)} valid results."
            }

    async def _combine_issues(self, 
                           chunk_results: List[Dict[str, Any]], 
                           document_info: Dict[str, Any]) -> Dict[str, Any]:
        """Combine issue identification results."""
        # Extract issues from all chunks
        all_issues = []
        for result in chunk_results:
            if "error" in result:
                continue
                
            # Handle different response formats
            if "issues" in result:
                issues = result["issues"]
                if isinstance(issues, list):
                    all_issues.extend(issues)
            # Look in raw output if no structured data
            elif "raw_output" in result:
                # Try to extract issues from text
                extracted = self._extract_json(result["raw_output"])
                if "issues" in extracted and isinstance(extracted["issues"], list):
                    all_issues.extend(extracted["issues"])
        
        # Deduplicate issues
        unique_issues = []
        seen_titles = set()
        
        for issue in all_issues:
            title = issue.get('title', '').lower()
            if title and title not in seen_titles:
                unique_issues.append(issue)
                seen_titles.add(title)
        
        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_issues = sorted(
            unique_issues,
            key=lambda x: severity_order.get(x.get('severity', '').lower(), 99)
        )
        
        # Generate a summary
        summary = f"Identified {len(sorted_issues)} issues across {len(chunk_results)} document sections."
        
        return {
            "pass_type": self.config['pass_type'],
            "purpose": self.config.get('purpose', ''),
            "issues": sorted_issues,
            "summary": summary
        }

    async def _combine_opportunities(self, 
                                  chunk_results: List[Dict[str, Any]], 
                                  document_info: Dict[str, Any]) -> Dict[str, Any]:
        """Combine opportunity identification results."""
        # Similar to _combine_issues but for opportunities
        all_opportunities = []
        for result in chunk_results:
            if "error" in result:
                continue
                
            if "opportunities" in result:
                opportunities = result["opportunities"]
                if isinstance(opportunities, list):
                    all_opportunities.extend(opportunities)
            elif "raw_output" in result:
                extracted = self._extract_json(result["raw_output"])
                if "opportunities" in extracted and isinstance(extracted["opportunities"], list):
                    all_opportunities.extend(extracted["opportunities"])
        
        # Deduplicate
        unique_opportunities = []
        seen_titles = set()
        
        for opportunity in all_opportunities:
            title = opportunity.get('title', '').lower()
            if title and title not in seen_titles:
                unique_opportunities.append(opportunity)
                seen_titles.add(title)
        
        # Sort by impact
        impact_order = {"transformative": 0, "high": 1, "medium": 2, "low": 3}
        sorted_opportunities = sorted(
            unique_opportunities,
            key=lambda x: impact_order.get(x.get('impact', '').lower(), 99)
        )
        
        # Generate a summary
        summary = f"Identified {len(sorted_opportunities)} opportunities across {len(chunk_results)} document sections."
        
        return {
            "pass_type": self.config['pass_type'],
            "purpose": self.config.get('purpose', ''),
            "opportunities": sorted_opportunities,
            "summary": summary
        }

    async def _combine_actions(self, 
                            chunk_results: List[Dict[str, Any]], 
                            document_info: Dict[str, Any]) -> Dict[str, Any]:
        """Combine action item results."""
        # Similar pattern to other combiners but for action items
        all_actions = []
        for result in chunk_results:
            if "error" in result:
                continue
                
            # Look for both "actions" and "action_items" keys
            for key in ["actions", "action_items"]:
                if key in result:
                    actions = result[key]
                    if isinstance(actions, list):
                        all_actions.extend(actions)
            
            # Try raw output as fallback
            if "raw_output" in result and not all_actions:
                extracted = self._extract_json(result["raw_output"])
                for key in ["actions", "action_items"]:
                    if key in extracted and isinstance(extracted[key], list):
                        all_actions.extend(extracted[key])
        
        # Deduplicate by similarity
        unique_actions = []
        seen_tasks = set()
        
        for action in all_actions:
            # Get task description (support multiple field names)
            task = None
            for field in ["task", "description", "action"]:
                if field in action and action[field]:
                    task = action[field].lower()
                    break
                    
            if task and task not in seen_tasks:
                unique_actions.append(action)
                seen_tasks.add(task)
        
        # Create summary
        summary = f"Extracted {len(unique_actions)} action items across {len(chunk_results)} document sections."
        
        # Group actions by assignee
        actions_by_assignee = {}
        unassigned_actions = []
        
        for action in unique_actions:
            assignee = action.get('assignee', '')
            if assignee and assignee.lower() not in ['none', 'n/a', 'unassigned']:
                if assignee not in actions_by_assignee:
                    actions_by_assignee[assignee] = []
                actions_by_assignee[assignee].append(action)
            else:
                unassigned_actions.append(action)
        
        return {
            "pass_type": self.config['pass_type'],
            "purpose": self.config.get('purpose', ''),
            "actions": unique_actions,
            "actions_by_assignee": actions_by_assignee,
            "unassigned_actions": unassigned_actions,
            "summary": summary
        }

# Update the create_pass_processor function
def create_pass_processor(pass_type: str, llm_client, options=None):
    """
    Factory function to create a pass processor.

    Args:
        pass_type: Type of pass to create
        llm_client: LLM client for text analysis
        options: Processing options

    Returns:
        Pass processor instance or None if not found
    """
    # Check for configuration file in passes/configurations directory
    config_dir = Path("passes/configurations")
    config_path = config_dir / f"{pass_type}.json"

    # Ensure directories exist
    if not config_dir.exists():
        try:
            logger.info(f"Creating missing configurations directory: {config_dir}")
            config_dir.mkdir(exist_ok=True, parents=True)
        except Exception as e:
            logger.error(f"Error creating configurations directory: {e}")
    
    # Check if configuration exists
    if not config_path.exists():
        # For issue_identification, create a default configuration if missing
        if pass_type == "issue_identification":
            try:
                logger.info(f"Creating default configuration for {pass_type}")
                default_config = {
                    "pass_type": "issue_identification",
                    "purpose": "Identify issues, problems, challenges, and concerns in the document",
                    "instructions": "Analyze the document to identify and describe issues or problems mentioned",
                    "process_by_chunks": False,
                    "prompt_template": "You are an expert analyst specializing in issue identification. Your task is to analyze the following document to identify any issues, problems, challenges, or concerns.\n\nDocument Summary:\n{summary}\n\nDocument Information:\n- Document Type: {document_info.get('is_meeting_transcript', False) and 'Meeting Transcript' or 'Document'}\n\nUser Instructions: {options.get('user_instructions', 'Identify all significant issues.')}\n\nFor each issue you identify, please provide:\n1. A clear, concise title\n2. A detailed description of the issue\n3. The severity level (critical, high, medium, or low)\n4. Who mentioned or is associated with the issue (if identifiable)\n5. Relevant context or direct quotes from the document\n\nReturn the results in JSON format:\n```json\n{\n  \"issues\": [\n    {\n      \"title\": \"Issue title\",\n      \"description\": \"Detailed description\",\n      \"severity\": \"critical|high|medium|low\",\n      \"speaker\": \"Person who mentioned it (if applicable)\",\n      \"context\": \"Relevant quote or context from the document\"\n    }\n  ],\n  \"summary\": \"Brief summary of the key issues identified\"\n}\n```\n\nFocus on the most significant issues first."
                }
                
                with open(config_path, 'w') as f:
                    import json
                    json.dump(default_config, f, indent=2)
                
                logger.info(f"Created default configuration at {config_path}")
            except Exception as e:
                logger.error(f"Error creating default configuration: {e}")
                return None
        else:
            logger.error(f"No configuration found for pass type: {pass_type}")
            return None

    if config_path.exists():
        logger.info(f"Creating pass processor for {pass_type} with config at {config_path}")
        try:
            return TemplatedPass(str(config_path), llm_client, options)
        except Exception as e:
            logger.error(f"Error creating pass processor: {e}")
            return None
    else:
        logger.error(f"Configuration file still not found after creation attempt: {config_path}")
        return None