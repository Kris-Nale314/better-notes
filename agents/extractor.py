"""
Extractor Agent - Specialized in identifying items in document chunks.
Clean implementation that leverages the new BaseAgent architecture.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base import BaseAgent

logger = logging.getLogger(__name__)

class ExtractorAgent(BaseAgent):
    """
    Agent specialized in extracting specific information from document chunks.
    Identifies issues and adds initial metadata for each issue.
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
    
    async def process(self, context):
        """
        Process chunks using the context.
        
        Args:
            context: ProcessingContext object
            
        Returns:
            Extraction results for all chunks
        """
        # Get chunks and metadata from context
        chunks = context.chunks
        chunk_metadata = context.chunk_metadata
        
        # Process each chunk
        extraction_results = []
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks):
            # Get metadata for this chunk
            metadata = chunk_metadata[i] if i < len(chunk_metadata) else {"index": i}
            
            # Extract from this chunk
            result = await self.extract_from_chunk(
                chunk=chunk,
                document_info=context.document_info,
                chunk_metadata=metadata
            )
            
            # Add to results
            extraction_results.append(result)
            
            # Update progress in context if available
            if hasattr(context, 'update_progress') and hasattr(context.metadata, 'get'):
                progress_base = 0.22  # Starting point for extraction
                progress_per_chunk = 0.28 / total_chunks  # 28% of total progress is for extraction
                progress = progress_base + (i + 1) * progress_per_chunk
                
                callback = context.metadata.get('progress_callback')
                if callback:
                    context.update_progress(
                        progress,
                        f"Extracted from chunk {i+1}/{total_chunks}",
                        callback
                    )
        
        return extraction_results
    
    async def extract_from_chunk(
        self, 
        chunk: str, 
        document_info: Optional[Dict[str, Any]] = None,
        chunk_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract information from a document chunk.
        
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
            "document_info": document_info,
            "chunk_metadata": chunk_metadata or {},
            "index": chunk_metadata.get("index", 0) if chunk_metadata else 0,
            "position": chunk_metadata.get("position", "unknown") if chunk_metadata else "unknown",
            "chunk_type": chunk_metadata.get("chunk_type", "unknown") if chunk_metadata else "unknown",
        }
        
        # Execute extraction
        result = await self.execute_task(extraction_context)
        
        # Enhance result with metadata
        return self._enhance_result_with_metadata(result, chunk_metadata)
    
    def _get_stage_specific_content(self, context) -> str:
        """Get stage-specific content for the prompt."""
        if isinstance(context, dict) and "document_chunk" in context:
            # Add document chunk as the main content
            content = f"DOCUMENT CHUNK:\n{context['document_chunk']}\n\n"
            
            # Add position information if available
            if "position" in context:
                position = context["position"]
                content += f"POSITION IN DOCUMENT: {position}\n"
                
                # Add position-specific guidance
                if position == "introduction":
                    content += "This is the introduction section. Look for initial mentions of problems or challenges.\n"
                elif position == "conclusion":
                    content += "This is the conclusion section. Look for unresolved issues or future considerations.\n"
            
            # Add chunk metadata if helpful
            if "chunk_type" in context:
                content += f"CHUNK TYPE: {context['chunk_type']}\n"
            
            if "index" in context:
                content += f"CHUNK INDEX: {context['index']}\n"
            
            return content
            
        return ""
    
    def _enhance_result_with_metadata(self, result: Any, chunk_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enhance the extraction result with additional metadata.
        
        Args:
            result: The extraction result
            chunk_metadata: Metadata about the chunk
            
        Returns:
            Enhanced result with metadata
        """
        # Handle string results
        if isinstance(result, str):
            try:
                # Try to parse as JSON
                parsed_result = self.parse_llm_json(result)
                if isinstance(parsed_result, dict):
                    result = parsed_result
                else:
                    # Create basic structure
                    result = {self._get_items_field_name(): [{"description": result}]}
            except:
                # Not valid JSON, create basic structure
                result = {self._get_items_field_name(): [{"description": result}]}
        
        # Handle non-dictionary results
        if not isinstance(result, dict):
            result = {self._get_items_field_name(): [{"description": str(result)}]}
        
        # Ensure items field exists
        items_field = self._get_items_field_name()
        if items_field not in result:
            result[items_field] = []
        
        # Add chunk metadata to each item
        items = result[items_field]
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    # Add chunk index if not present
                    if "chunk_index" not in item and chunk_metadata:
                        item["chunk_index"] = chunk_metadata.get("index", 0)
                    
                    # Add position context if not present
                    if "location_context" not in item and chunk_metadata:
                        position = chunk_metadata.get("position", "unknown")
                        item["location_context"] = f"{position} section"
                    
                    # Add keywords if not present
                    if "keywords" not in item and "description" in item:
                        item["keywords"] = self._extract_keywords(item["description"])
        
        # Add extraction metadata
        result["_metadata"] = {
            "chunk_index": chunk_metadata.get("index", 0) if chunk_metadata else 0,
            "position": chunk_metadata.get("position", "unknown") if chunk_metadata else "unknown",
            "chunk_type": chunk_metadata.get("chunk_type", "unknown") if chunk_metadata else "unknown",
            "item_count": len(items) if isinstance(items, list) else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
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
            "risks": "risks"
        }
        
        return field_map.get(self.crew_type, f"{self.crew_type}_items")
    
    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """
        Extract key keywords from text.
        
        Args:
            text: Text to analyze
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction implementation
        import re
        from collections import Counter
        
        # Tokenize and clean
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stopwords
        stopwords = {"the", "and", "to", "of", "a", "in", "that", "it", "with", 
                    "for", "on", "is", "was", "be", "this", "are", "as", "but"}
        filtered_words = [word for word in words if word not in stopwords]
        
        # Count frequencies
        word_counts = Counter(filtered_words)
        
        # Get most common words
        return [word for word, _ in word_counts.most_common(max_keywords)]