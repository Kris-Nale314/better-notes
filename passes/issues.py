"""
Issue identification module for Better Notes.
Provides specialized handling for issue identification passes.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path

from lean.async_openai_adapter import AsyncOpenAIAdapter
from passes.passes import create_pass_processor, TemplatedPass

logger = logging.getLogger(__name__)

class IssueIdentifier:
    """
    Specialized handler for issue identification in documents.
    Uses the passes system for core functionality but adds issue-specific
    processing and formatting.
    """
    
    def __init__(self, llm_client, options=None):
        """
        Initialize the issue identifier.
        
        Args:
            llm_client: LLM client for text analysis
            options: Processing options
        """
        self.llm_client = llm_client
        self.options = options
        
        # Load configuration first to ensure it exists
        self.config = self._load_config()
        
        # Check if configurations directory exists
        config_dir = Path("passes/configurations")
        if not config_dir.exists():
            logger.warning(f"Creating missing configurations directory: {config_dir}")
            config_dir.mkdir(exist_ok=True, parents=True)
            
        # Check if issue_identification.json exists
        config_path = config_dir / "issue_identification.json"
        if not config_path.exists():
            logger.warning(f"Issue identification configuration not found at {config_path}")
            # If config is loaded but file doesn't exist, save it
            if self.config:
                try:
                    with open(config_path, 'w') as f:
                        json.dump(self.config, f, indent=2)
                    logger.info(f"Created issue identification configuration at {config_path}")
                except Exception as e:
                    logger.error(f"Error creating configuration file: {e}")
        
        # Create the underlying pass processor
        self.processor = create_pass_processor("issue_identification", llm_client, options)
        
        # Fallback if processor creation fails
        if not self.processor and self.config:
            logger.warning("Falling back to direct TemplatedPass creation")
            from passes.passes import TemplatedPass
            try:
                self.processor = TemplatedPass(str(config_path), llm_client, options)
                logger.info("Successfully created pass processor using fallback method")
            except Exception as e:
                logger.error(f"Fallback processor creation failed: {e}")
                self.processor = None
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load issue identification configuration.
        
        Returns:
            Configuration dictionary
        """
        # Try to load from file first
        config_path = Path("passes/configurations/issue_identification.json")
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    logger.info(f"Loading issue configuration from {config_path}")
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading issue configuration from file: {e}")
                # Fall through to default configuration
                
        # Default configuration if file not found or error loading
        logger.info("Using default issue identification configuration")
        return {
            "pass_type": "issue_identification",
            "purpose": "Identify issues, problems, challenges, and concerns in the document",
            "instructions": "Analyze the document to identify and describe issues or problems mentioned",
            "process_by_chunks": False,
            "prompt_template": """
            You are an expert analyst specializing in issue identification. Your task is to analyze the following document to identify any issues, problems, challenges, or concerns.

            Document Summary:
            {summary}

            Document Information:
            - Document Type: {document_info.get('is_meeting_transcript', False) and 'Meeting Transcript' or 'Document'}

            User Instructions: {options.get('user_instructions', 'Identify all significant issues.')}

            For each issue you identify, please provide:
            1. A clear, concise title
            2. A detailed description of the issue
            3. The severity level (critical, high, medium, or low)
            4. Who mentioned or is associated with the issue (if identifiable)
            5. Relevant context or direct quotes from the document

            CRITICAL issues are those that:
            - Represent immediate threats to operations, security, or compliance
            - Could cause significant financial loss or reputational damage
            - Are blocking major processes or deliverables

            HIGH severity issues are those that:
            - Significantly impact effectiveness or efficiency
            - Require substantial resources to address
            - Will likely escalate if not addressed soon

            MEDIUM severity issues are those that:
            - Cause ongoing inefficiency or limitations
            - Negatively impact some stakeholders
            - Should be addressed, but aren't urgent

            LOW severity issues are those that:
            - Represent minor inconveniences or concerns
            - Have minimal impact on operations or outcomes
            - Could be addressed through regular maintenance or improvements

            Return the results in JSON format:
            ```json
            {
              "issues": [
                {
                  "title": "Issue title",
                  "description": "Detailed description",
                  "severity": "critical|high|medium|low",
                  "speaker": "Person who mentioned it (if applicable)",
                  "context": "Relevant quote or context from the document"
                }
              ],
              "summary": "Brief summary of the key issues identified"
            }
            ```

            Focus on the most significant issues first. Be thorough but avoid inventing issues not supported by the document.
            """
        }
    
    async def process_document(self,
                               document_text: str,
                               document_info: Optional[Dict[str, Any]] = None,
                               progress_callback: Optional[Callable] = None,
                               prior_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a document for issue identification.
        
        Args:
            document_text: Document text
            document_info: Document metadata
            progress_callback: Progress callback function
            prior_result: Prior processing results
            
        Returns:
            Issue identification results
        """
        if not self.processor:
            logger.error("Issue identification processor not initialized")
            return {"error": "Issue identification processor not initialized"}
            
        # Process document using underlying pass processor
        if progress_callback:
            progress_callback(0.1, "Starting issue identification...")
            
        raw_result = await self.processor.process_document(
            document_text=document_text,
            document_info=document_info,
            progress_callback=progress_callback,
            prior_result=prior_result
        )
        
        if progress_callback:
            progress_callback(0.9, "Formatting issue results...")
            
        # Extract and format results
        formatted_result = self._format_results(raw_result)
        
        if progress_callback:
            progress_callback(1.0, "Issue identification complete")
            
        return formatted_result
    
    def _format_results(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format issue identification results.
        
        Args:
            raw_result: Raw results from pass processor
            
        Returns:
            Formatted results
        """
        # Extract issues from result
        issues = []
        
        if "result" in raw_result:
            result_data = raw_result["result"]
            if "issues" in result_data and isinstance(result_data["issues"], list):
                issues = result_data["issues"]
                
        # Sort issues by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_issues = sorted(
            issues,
            key=lambda x: severity_order.get(x.get("severity", "").lower(), 99)
        )
        
        # Apply any filtering from options
        if self.options and hasattr(self.options, 'get_pass_options'):
            pass_options = self.options.get_pass_options("issue_identification")
            if "severity_threshold" in pass_options:
                threshold = pass_options["severity_threshold"].lower()
                threshold_value = severity_order.get(threshold, 3)
                
                # Filter issues based on threshold
                sorted_issues = [
                    issue for issue in sorted_issues
                    if severity_order.get(issue.get("severity", "").lower(), 99) <= threshold_value
                ]
        
        # Group issues by severity
        issues_by_severity = {}
        for issue in sorted_issues:
            severity = issue.get("severity", "medium").lower()
            if severity not in issues_by_severity:
                issues_by_severity[severity] = []
            issues_by_severity[severity].append(issue)
        
        # Create formatted output if format_template is available
        formatted_output = ""
        if "format_template" in self.config:
            try:
                formatted_output = self._apply_format_template(
                    self.config["format_template"],
                    {
                        "issues": sorted_issues,
                        "summary": result_data.get("summary", "")
                    }
                )
            except Exception as e:
                logger.error(f"Error formatting issues: {e}")
        
        # Create result with enhanced metadata
        return {
            "issues": sorted_issues,
            "issues_by_severity": issues_by_severity,
            "summary": result_data.get("summary", ""),
            "formatted_output": formatted_output,
            "total_issues": len(sorted_issues),
            "critical_issues": len(issues_by_severity.get("critical", [])),
            "high_issues": len(issues_by_severity.get("high", [])),
            "medium_issues": len(issues_by_severity.get("medium", [])),
            "low_issues": len(issues_by_severity.get("low", []))
        }
        
    def _apply_format_template(self, template: str, data: Dict[str, Any]) -> str:
        """
        Apply a format template to data.
        
        Args:
            template: Template string
            data: Data to format
            
        Returns:
            Formatted string
        """
        # Simple template rendering implementation
        # Replace variables: {{variable}}
        import re
        
        result = template
        
        # Replace simple variables
        for variable in re.findall(r'{{([^{]+?)}}', result):
            var_name = variable.strip()
            # Handle nested paths with dot notation
            parts = var_name.split('.')
            
            # Get value from data
            value = data
            try:
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        value = ""
                        break
                # Replace in template
                result = result.replace('{{' + variable + '}}', str(value))
            except Exception:
                # Replace with empty string if error
                result = result.replace('{{' + variable + '}}', "")
        
        # Handle for loops: {% for item in items %} ... {% endfor %}
        for loop_match in re.finditer(
            r'{%\s*for\s+(\w+)\s+in\s+(\w+)\s*%}(.*?){%\s*endfor\s*%}',
            result,
            re.DOTALL
        ):
            loop_var, collection_name, loop_content = loop_match.groups()
            
            # Get the collection
            if collection_name in data and isinstance(data[collection_name], list):
                collection = data[collection_name]
                
                # Process loop content for each item
                rendered_items = []
                for item in collection:
                    item_content = loop_content
                    
                    # Replace item variables
                    for var_match in re.finditer(r'{{' + loop_var + r'\.(\w+)}}', item_content):
                        item_prop = var_match.group(1)
                        if item_prop in item:
                            item_content = item_content.replace(
                                '{{' + loop_var + '.' + item_prop + '}}',
                                str(item[item_prop])
                            )
                    
                    # Handle conditionals within the loop
                    for cond_match in re.finditer(
                        r'{%\s*if\s+' + loop_var + r'\.(\w+)\s*%}(.*?){%\s*endif\s*%}',
                        item_content,
                        re.DOTALL
                    ):
                        cond_prop, cond_content = cond_match.groups()
                        if cond_prop in item and item[cond_prop]:
                            # Condition is true, keep content
                            item_content = item_content.replace(
                                '{% if ' + loop_var + '.' + cond_prop + ' %}' + cond_content + '{% endif %}',
                                cond_content
                            )
                        else:
                            # Condition is false, remove content
                            item_content = item_content.replace(
                                '{% if ' + loop_var + '.' + cond_prop + ' %}' + cond_content + '{% endif %}',
                                ''
                            )
                    
                    rendered_items.append(item_content)
                
                # Replace the entire loop with rendered items
                loop_replacement = ''.join(rendered_items)
                result = result.replace(
                    '{% for ' + loop_var + ' in ' + collection_name + ' %}' + loop_content + '{% endfor %}',
                    loop_replacement
                )
            else:
                # Collection not found, remove the loop
                result = result.replace(
                    '{% for ' + loop_var + ' in ' + collection_name + ' %}' + loop_content + '{% endfor %}',
                    ''
                )
        
        return result
    
    def process_document_sync(self,
                             document_text: str,
                             document_info: Optional[Dict[str, Any]] = None,
                             progress_callback: Optional[Callable] = None,
                             prior_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a document synchronously.
        
        Args:
            document_text: Document text
            document_info: Document metadata
            progress_callback: Progress callback function
            prior_result: Prior processing results
            
        Returns:
            Issue identification results
        """
        # Create event loop
        loop = asyncio.new_event_loop()
        try:
            # Run async function
            return loop.run_until_complete(
                self.process_document(
                    document_text=document_text,
                    document_info=document_info,
                    progress_callback=progress_callback,
                    prior_result=prior_result
                )
            )
        finally:
            # Clean up
            loop.close()


def create_issue_identifier(llm_client, options=None) -> IssueIdentifier:
    """
    Create an issue identifier instance.
    
    Args:
        llm_client: LLM client for text analysis
        options: Processing options
        
    Returns:
        Issue identifier instance
    """
    return IssueIdentifier(llm_client, options)