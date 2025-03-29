"""
Simplified Extractor Agent for Better Notes.
Focuses on extracting as many issues as possible with minimal complexity.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base import BaseAgent

logger = logging.getLogger(__name__)

class ExtractorAgent(BaseAgent):
    """
    Simplified Extractor agent that focuses on finding as many potential issues as possible.
    Keeps things simple and ensures all extracted issues have the necessary fields.
    """
    
    def __init__(
        self,
        llm_client,
        crew_type: str,
        config: Optional[Dict[str, Any]] = None,
        config_manager = None,
        verbose: bool = True,
        max_chunk_size: int = 1500,
        max_rpm: int = 10
    ):
        """Initialize an extractor agent."""
        super().__init__(
            llm_client=llm_client,
            agent_type="extractor",
            crew_type=crew_type,
            config=config,
            config_manager=config_manager,
            verbose=verbose,
            max_chunk_size=max_chunk_size,
            max_rpm=max_rpm
        )
        
        logger.info(f"Simplified ExtractorAgent initialized for {crew_type}")
    
    async def process(self, context):
        """
        Process chunks using the context with straightforward approach.
        
        Args:
            context: ProcessingContext object
            
        Returns:
            Extraction results for all chunks
        """
        logger.info("ExtractorAgent starting extraction process")
        
        # Get chunks and metadata from context
        chunks = context.chunks
        chunk_metadata = context.chunk_metadata
        
        if not chunks:
            logger.warning("No chunks found in context for extraction")
            return []
        
        # Process each chunk
        extraction_results = []
        total_chunks = len(chunks)
        total_issues_found = 0
        
        logger.info(f"Processing {total_chunks} chunks for extraction")
        
        for i, chunk in enumerate(chunks):
            # Get metadata for this chunk
            metadata = chunk_metadata[i] if i < len(chunk_metadata) else {"index": i}
            
            try:
                # Extract from this chunk
                result = await self.extract_from_chunk(
                    chunk=chunk,
                    document_info=context.document_info,
                    chunk_metadata=metadata
                )
                
                # Count issues
                issues_field = self._get_items_field_name()
                if issues_field in result and isinstance(result[issues_field], list):
                    issues_found = len(result[issues_field])
                    total_issues_found += issues_found
                    logger.info(f"Chunk {i+1}/{total_chunks}: Found {issues_found} issues")
                
                # Add to results
                extraction_results.append(result)
                
                logger.info(f"Successfully extracted from chunk {i+1}/{total_chunks}")
                
            except Exception as e:
                # Log error but continue with other chunks
                logger.error(f"Error extracting from chunk {i+1}/{total_chunks}: {e}")
                
                # Add empty result for this chunk
                extraction_results.append({
                    "error": str(e),
                    "chunk_index": i,
                    self._get_items_field_name(): []  # Empty items list
                })
            
            # Update progress in context if available
            if hasattr(context, 'update_progress') and callable(getattr(context, 'update_progress', None)):
                progress_base = 0.22  # Starting point for extraction
                progress_per_chunk = 0.28 / total_chunks  # 28% of total progress is for extraction
                progress = progress_base + (i + 1) * progress_per_chunk
                
                callback = getattr(context.metadata, 'get', lambda x: None)('progress_callback')
                if callback:
                    context.update_progress(
                        progress,
                        f"Extracted from chunk {i+1}/{total_chunks} - Found {total_issues_found} issues so far",
                        callback
                    )
        
        logger.info(f"Completed extraction of {total_chunks} chunks, found {total_issues_found} total issues")
        
        # Track issue count if context supports it
        if hasattr(context, 'track_issue_count'):
            context.track_issue_count('extraction', total_issues_found)
        
        return extraction_results
    
    async def extract_from_chunk(
        self, 
        chunk: str, 
        document_info: Optional[Dict[str, Any]] = None,
        chunk_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract issues from a document chunk with straightforward approach.
        
        Args:
            chunk: Text chunk to analyze
            document_info: Optional document metadata
            chunk_metadata: Optional metadata about the chunk
            
        Returns:
            Extraction results
        """
        # Ensure chunk size is within limits
        safe_chunk = self.truncate_text(chunk, self.max_chunk_size)
        
        # Create extraction context
        extraction_context = {
            "document_chunk": safe_chunk,
            "chunk_metadata": chunk_metadata or {},
            "position": chunk_metadata.get("position", "unknown") if chunk_metadata else "unknown",
            "chunk_index": chunk_metadata.get("index", 0) if chunk_metadata else 0
        }
        
        # Execute extraction
        result = await self.execute_task(extraction_context)
        
        # Parse result if it's a string
        if isinstance(result, str):
            try:
                result = self.parse_llm_json(result)
            except Exception as e:
                logger.warning(f"Error parsing extraction result: {e}")
                
                # Create a basic structure with the text as description
                issues_field = self._get_items_field_name()
                return {
                    issues_field: [{
                        "title": "Parsing Error",
                        "description": result[:500] + ("..." if len(result) > 500 else ""),
                        "severity": "medium",
                        "category": "technical"
                    }],
                    "chunk_index": chunk_metadata.get("index", 0) if chunk_metadata else 0
                }
        
        # Ensure result is a dictionary
        if not isinstance(result, dict):
            issues_field = self._get_items_field_name()
            result = {issues_field: []}
        
        # Ensure issues field exists
        issues_field = self._get_items_field_name()
        if issues_field not in result:
            result[issues_field] = []
        
        # Ensure each issue has required fields
        self._ensure_required_fields(result[issues_field], chunk_metadata)
        
        # Add chunk metadata
        result["chunk_index"] = chunk_metadata.get("index", 0) if chunk_metadata else 0
        result["position"] = chunk_metadata.get("position", "unknown") if chunk_metadata else "unknown"
        
        return result
    
    def _get_stage_specific_content(self, context) -> str:
        """
        Get stage-specific content for the prompt.
        
        Args:
            context: Extraction context
            
        Returns:
            Stage-specific content string
        """
        if isinstance(context, dict) and "document_chunk" in context:
            # Add document chunk as the main content
            content = f"""
DOCUMENT CHUNK TO ANALYZE:
{context['document_chunk']}

YOUR TASK:
Extract ALL potential issues, problems, challenges, risks, or concerns from this document chunk.
Be thorough and comprehensive - it's better to identify too many issues than to miss important ones.

For each issue you identify, provide:
1. A clear, concise title
2. A detailed description of the issue
3. A severity assessment (critical, high, medium, or low)
4. The best category that fits from: technical, process, resource, quality, risk, compliance

CHUNK CONTEXT:
Position in document: {context.get('position', 'unknown')}
Chunk index: {context.get('chunk_index', 0)}

IMPORTANT GUIDELINES:
- Be comprehensive - identify as many issues as possible
- Include issues that are explicitly mentioned AND those that are implied
- Prioritize concrete, specific issues over vague or general concerns
- Consider perspectives of different stakeholders (customers, employees, management)
- Look for risks, challenges, bottlenecks, inefficiencies, and problems
- Don't filter out issues you think might be unimportant - include everything
- Ensure every issue has a title and description at minimum

OUTPUT FORMAT:
Provide a JSON object with an "issues" array containing all identified issues.
Each issue should have: title, description, severity, and category fields.
"""
            return content
        
        return ""
    
    def _ensure_required_fields(self, issues: List[Dict[str, Any]], chunk_metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Ensure all issues have the required fields, adding defaults if missing.
        
        Args:
            issues: List of issues to check
            chunk_metadata: Metadata about the chunk
        """
        if not isinstance(issues, list):
            return
        
        for i, issue in enumerate(issues):
            if not isinstance(issue, dict):
                continue
            
            # Ensure title exists
            if "title" not in issue or not issue["title"]:
                issue["title"] = f"Untitled Issue {i+1}"
            
            # Ensure description exists
            if "description" not in issue or not issue["description"]:
                issue["description"] = issue.get("title", "No description provided")
            
            # Ensure severity exists
            if "severity" not in issue or not issue["severity"]:
                issue["severity"] = "medium"
            
            # Ensure category exists
            if "category" not in issue or not issue["category"]:
                issue["category"] = self._infer_category(issue.get("description", ""), issue.get("title", ""))
            
            # Add chunk metadata
            if chunk_metadata:
                issue["chunk_index"] = chunk_metadata.get("index", 0)
                issue["location_context"] = f"{chunk_metadata.get('position', 'unknown')} section"
    
    def _get_items_field_name(self) -> str:
        """
        Get the field name for extracted items based on crew type.
        
        Returns:
            Field name for items
        """
        # Map crew types to field names
        field_map = {
            "issues": "issues",
            "actions": "action_items",
            "opportunities": "opportunities",
            "risks": "risks",
            "insights": "insights"
        }
        
        return field_map.get(self.crew_type, f"{self.crew_type}_items")
    
    def _infer_category(self, description: str, title: str = "") -> str:
        """
        Infer a category for an issue based on its description and title.
        Simplified version that uses basic keyword matching.
        
        Args:
            description: Issue description
            title: Issue title
            
        Returns:
            Inferred category
        """
        if self.crew_type != "issues":
            return ""
            
        # Get available categories from config
        available_categories = []
        if "issue_definition" in self.config and "categories" in self.config["issue_definition"]:
            available_categories = self.config["issue_definition"]["categories"]
        
        # Default categories if not in config
        if not available_categories:
            available_categories = ["technical", "process", "resource", "quality", "risk", "compliance"]
        
        # Combine text for analysis
        text = (title + " " + description).lower()
        
        # Simple keyword mapping to categories
        category_keywords = {
            "technical": ["technical", "technology", "system", "software", "hardware", "infrastructure", "bug"],
            "process": ["process", "procedure", "workflow", "approach", "steps", "method"],
            "resource": ["resource", "budget", "cost", "funding", "staff", "personnel", "time", "money"],
            "quality": ["quality", "standard", "performance", "metric", "test"],
            "risk": ["risk", "threat", "danger", "security", "mitigation"],
            "compliance": ["compliance", "regulation", "requirement", "legal", "law", "rule"]
        }
        
        # Check each category
        for category, keywords in category_keywords.items():
            if category in available_categories:
                for keyword in keywords:
                    if keyword in text:
                        return category
        
        # Default to the first available category
        return available_categories[0] if available_categories else "general"