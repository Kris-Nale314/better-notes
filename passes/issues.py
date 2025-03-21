"""
Issue identification module for Better Notes.
Initializes issue identification pass processor.
"""

import logging
import asyncio
import json
import re
from typing import Dict, Any, Optional, Callable
from pathlib import Path

from passes.passes import create_pass_processor, TemplatedPass

logger = logging.getLogger(__name__)

def create_issue_identifier(llm_client, options=None):
    """
    Create an issue identification pass processor.
    
    This factory function creates a pass processor for issue identification
    using the configuration from passes/configurations/issue_identification.json.
    
    Args:
        llm_client: LLM client for text processing
        options: Processing options
        
    Returns:
        Configured pass processor for issue identification
    """
    logger.info("Creating issue identification pass processor")
    
    # Ensure configuration directory exists
    config_dir = Path("passes/configurations")
    config_path = config_dir / "issue_identification.json"
    
    # Log configuration status
    if config_path.exists():
        logger.info(f"Found issue identification configuration at {config_path}")
    else:
        logger.warning(f"Configuration file not found: {config_path}")
        
    # Create the pass processor
    processor = create_pass_processor("issue_identification", llm_client, options)
    
    # Fallback 1: Try direct instantiation if the factory fails
    if processor is None and config_path.exists():
        logger.warning("Factory creation failed, trying direct instantiation")
        try:
            processor = TemplatedPass(str(config_path), llm_client, options)
            logger.info("Successfully created processor through direct instantiation")
        except Exception as e:
            logger.error(f"Direct instantiation failed: {e}")
            processor = None
    
    # Fallback 2: Create a basic pass processor if all else fails
    if processor is None:
        logger.warning("All standard creation methods failed, using fallback processor")
        processor = FallbackIssueProcessor(llm_client, options)
        logger.info("Created fallback processor")
    
    return processor

class FallbackIssueProcessor:
    """
    Fallback processor for issue identification when normal initialization fails.
    Implements the minimal interface needed for compatibility.
    """
    
    def __init__(self, llm_client, options=None):
        """
        Initialize fallback processor.
        
        Args:
            llm_client: LLM client for text analysis
            options: Processing options
        """
        self.llm_client = llm_client
        self.options = options
        logger.info("Initialized fallback issue processor")
    
    async def process_document(self, 
                             document_text: str,
                             document_info: Optional[Dict[str, Any]] = None,
                             progress_callback: Optional[Callable] = None,
                             prior_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a document to identify issues.
        
        Args:
            document_text: Document text
            document_info: Document metadata
            progress_callback: Progress callback function
            prior_result: Prior processing results
            
        Returns:
            Dictionary with issues identified
        """
        if progress_callback:
            progress_callback(0.1, "Using fallback issue processor...")
        
        # Extract user instructions
        user_instructions = ""
        if self.options and hasattr(self.options, 'user_instructions'):
            user_instructions = self.options.user_instructions or ""
        
        # Create a basic but effective prompt
        prompt = f"""
        Analyze this document to identify key issues, problems, and challenges:

        {document_text[:8000]}  # Truncated to avoid token limits

        {f"User Instructions: {user_instructions}" if user_instructions else ""}

        For each issue, provide:
        1. A clear, concise title
        2. A detailed description
        3. Severity (critical, high, medium, low)
        4. Who mentioned the issue (if applicable)
        5. Context from the document supporting this issue

        Format as JSON:
        ```json
        {{
          "issues": [
            {{
              "title": "Issue title",
              "description": "Detailed description",
              "severity": "critical|high|medium|low",
              "speaker": "Person who mentioned it (if applicable)",
              "context": "Supporting text from document"
            }}
          ],
          "summary": "Brief summary of key issues identified"
        }}
        ```
        """
        
        if progress_callback:
            progress_callback(0.3, "Analyzing document for issues...")
            
        try:
            # Get analysis from LLM
            response = await self.llm_client.generate_completion_async(prompt)
            
            if progress_callback:
                progress_callback(0.8, "Processing results...")
            
            # Try to extract JSON
            result = self._extract_json(response)
            
            if progress_callback:
                progress_callback(0.9, "Formatting results...")
            
            # Return in the format expected by the UI
            return {"result": result}
            
        except Exception as e:
            logger.error(f"Error in fallback processing: {e}")
            if progress_callback:
                progress_callback(1.0, f"Error: {str(e)}")
            
            # Return minimal valid result
            return {
                "result": {
                    "issues": [],
                    "summary": f"Error processing document: {str(e)}",
                    "error": str(e)
                }
            }
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON from text response.
        
        Args:
            text: Response text
            
        Returns:
            Extracted JSON or default structure
        """
        # Try to find JSON in code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from code block")
        
        # Try to find JSON without code blocks (looser match)
        try:
            # Look for an object starting with {"issues":
            issues_match = re.search(r'\{\s*"issues"\s*:\s*\[', text)
            if issues_match:
                # Find the start of this object
                start_idx = text.rfind('{', 0, issues_match.start() + 1)
                if start_idx == -1:
                    start_idx = issues_match.start()
                
                # Try to parse from this position
                import json5  # More forgiving JSON parser
                remaining = text[start_idx:]
                try:
                    return json.loads(remaining)
                except json.JSONDecodeError:
                    # Try with json5 if available
                    try:
                        return json5.loads(remaining)
                    except:
                        pass
        except:
            pass
        
        # If JSON extraction failed, try to extract issues manually
        issues = []
        
        # Look for issue patterns
        issue_blocks = re.findall(r'(Issue|Problem|Challenge)[\s\d]*:([^\n]+)', text, re.IGNORECASE)
        
        for i, (_, title) in enumerate(issue_blocks):
            title = title.strip()
            if title:
                issues.append({
                    "title": title,
                    "description": f"Issue extracted from text (details not structured)",
                    "severity": "medium"  # Default severity
                })
        
        # If that didn't work, last resort is to treat the entire response as one issue
        if not issues and len(text.strip()) > 20:
            issues.append({
                "title": "Extracted Issue",
                "description": text[:500] + ("..." if len(text) > 500 else ""),
                "severity": "medium"
            })
        
        return {
            "issues": issues,
            "summary": "Issues extracted using fallback processor"
        }
    
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
            Dictionary with issues identified
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