# agents/extractor.py
"""
Extractor Agent - Specialized in identifying items in document chunks.
Works with ProcessingContext and integrated crew architecture.
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
    Adds metadata to extracted items and works with ProcessingContext.
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
        """
        Initialize an extractor agent.
        
        Args:
            llm_client: LLM client for agent communication
            crew_type: Type of crew (issues, actions, opportunities)
            config: Optional pre-loaded configuration
            config_manager: Optional config manager
            verbose: Whether to enable verbose mode
            max_chunk_size: Maximum size of text chunks to process
            max_rpm: Maximum requests per minute
        """
        super().__init__(
            llm_client=llm_client,
            agent_type="extraction",
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
        for i, chunk in enumerate(chunks):
            # Get metadata for this chunk
            metadata = chunk_metadata[i] if i < len(chunk_metadata) else {"index": i}
            
            # Extract from this chunk
            result = self.extract_from_chunk(
                chunk=chunk,
                document_info=context.document_info,
                chunk_metadata=metadata
            )
            
            # Add to results
            extraction_results.append(result)
            
            # Update progress in context if available
            if hasattr(context, 'update_progress') and context.metadata.get('progress_callback'):
                progress_base = 0.22  # Starting point for extraction
                progress_per_chunk = 0.28 / len(chunks)  # 28% of total progress is for extraction
                progress = progress_base + (i + 1) * progress_per_chunk
                context.update_progress(
                    progress,
                    f"Extracted from chunk {i+1}/{len(chunks)}",
                    context.metadata.get('progress_callback')
                )
        
        return extraction_results
    
    # In agents/extractor.py:
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
        
        # Prepare context with metadata
        context = {
            "document_chunk": safe_chunk,
            "document_info": self._extract_relevant_info(document_info)
        }
        
        # Add chunk metadata if available
        if chunk_metadata:
            # Add position-specific guidance
            if "position" in chunk_metadata:
                position = chunk_metadata["position"]
                position_guidance = self._get_position_guidance(position)
                if position_guidance:
                    context["position_guidance"] = position_guidance
            
            # Add chunk type information
            if "chunk_type" in chunk_metadata:
                chunk_type = chunk_metadata["chunk_type"]
                context["chunk_type"] = chunk_type
                
                # Add chunk type guidance
                type_guidance = self._get_chunk_type_guidance(chunk_type)
                if type_guidance:
                    context["type_guidance"] = type_guidance
            
            # Add index information
            if "index" in chunk_metadata:
                context["chunk_index"] = chunk_metadata["index"]
        
        # Execute the extraction task - ADD AWAIT HERE
        result = await self.execute_task(context=context)
        
        # Enhance the result with chunk metadata
        result = self._enhance_result_with_metadata(result, chunk_metadata)
        
        return result
    
    def _get_stage_specific_content(self, context) -> str:
        """Get stage-specific content for the prompt."""
        if not hasattr(context, 'document_chunk'):
            return ""
            
        # Add document chunk
        content = f"DOCUMENT CHUNK:\n{context['document_chunk']}\n\n"
        
        # Add position guidance if available
        if 'position_guidance' in context:
            content += f"POSITION GUIDANCE:\n{context['position_guidance']}\n\n"
        
        # Add chunk type guidance if available
        if 'type_guidance' in context:
            content += f"CHUNK TYPE GUIDANCE:\n{context['type_guidance']}\n\n"
        
        # Add chunk index if available
        if 'chunk_index' in context:
            content += f"CHUNK INDEX: {context['chunk_index']}\n\n"
            
        return content
    
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
                parsed_result = json.loads(result)
                if isinstance(parsed_result, dict):
                    result = parsed_result
            except json.JSONDecodeError:
                # Not JSON, convert to basic dictionary
                key_field = self._get_items_field_name()
                result = {
                    key_field: [
                        {"description": result, "chunk_index": chunk_metadata.get("index", 0) if chunk_metadata else 0}
                    ]
                }
        
        # Handle non-dictionary results
        if not isinstance(result, dict):
            key_field = self._get_items_field_name()
            result = {
                key_field: [
                    {"description": str(result), "chunk_index": chunk_metadata.get("index", 0) if chunk_metadata else 0}
                ]
            }
        
        # Ensure the key field exists in the result
        key_field = self._get_items_field_name()
        if key_field not in result:
            result[key_field] = []
        
        # Add chunk metadata to each item
        items = result[key_field]
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    # Add chunk index if not present
                    if "chunk_index" not in item and chunk_metadata:
                        item["chunk_index"] = chunk_metadata.get("index", 0)
                    
                    # Extract keywords if not present
                    if "keywords" not in item and "description" in item:
                        item["keywords"] = self.extract_keywords(item["description"])
                    
                    # Add location context if not present
                    if "location_context" not in item and chunk_metadata:
                        position = chunk_metadata.get("position", "unknown")
                        item["location_context"] = f"{position} section"
        
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
            Field name (e.g., "issues", "action_items", etc.)
        """
        # Default mapping
        field_mapping = {
            "issues": "issues",
            "actions": "action_items",
            "opportunities": "opportunities",
            "risks": "risks"
        }
        
        # Get from config if available
        field_name = field_mapping.get(self.crew_type, f"{self.crew_type}_items")
        
        return field_name
    
    def _extract_relevant_info(self, document_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract only the most relevant information from document_info to avoid token waste.
        
        Args:
            document_info: Full document information
            
        Returns:
            Dictionary with only the most relevant information
        """
        if not document_info:
            return {}
        
        relevant = {}
        
        # Copy only the most relevant fields
        relevant_fields = [
            "is_meeting_transcript", 
            "client_name", 
            "meeting_purpose"
        ]
        
        for field in relevant_fields:
            if field in document_info:
                relevant[field] = document_info[field]
        
        # Add a few key topics if available
        if "preview_analysis" in document_info and isinstance(document_info["preview_analysis"], dict):
            preview = document_info["preview_analysis"]
            if "key_topics" in preview:
                relevant["key_topics"] = preview["key_topics"][:3]  # Only include top 3
        
        return relevant
    
    def _get_position_guidance(self, position: str) -> str:
        """
        Get position-specific guidance for extraction.
        
        Args:
            position: Position in document (introduction, conclusion, etc.)
            
        Returns:
            Guidance string for this position
        """
        guidance = {
            "introduction": f"This is the INTRODUCTION of the document. Look for early mentions of {self.crew_type} or potential challenges.",
            "conclusion": f"This is the CONCLUSION of the document. Look for unresolved {self.crew_type} or future considerations.",
            "early": f"This is an EARLY section of the document. Focus on initial mentions of {self.crew_type}.",
            "middle": f"This is a MIDDLE section of the document. Look for detailed descriptions of {self.crew_type}.",
            "late": f"This is a LATE section of the document. Watch for remaining {self.crew_type} that need attention.",
            "full_document": f"This is the FULL DOCUMENT. Look for {self.crew_type} across the entire content."
        }
        
        return guidance.get(position, "")
    
    def _get_chunk_type_guidance(self, chunk_type: str) -> str:
        """
        Get chunk type-specific guidance for extraction.
        
        Args:
            chunk_type: Type of chunk (transcript_segment, structural_segment, etc.)
            
        Returns:
            Guidance string for this chunk type
        """
        guidance = {
            "transcript_segment": f"This is a TRANSCRIPT segment. Watch for {self.crew_type} mentioned in conversation.",
            "structural_segment": f"This is a STRUCTURAL segment. Pay attention to sections that highlight {self.crew_type}.",
            "content_segment": f"This is a general CONTENT segment. Look for descriptions of {self.crew_type}.",
            "fixed_size_segment": f"This is a FIXED-SIZE segment. Note any partial {self.crew_type} that might continue in other chunks."
        }
        
        return guidance.get(chunk_type, "")
    
    def extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """
        Extract key keywords from text for better metadata.
        
        Args:
            text: Text to analyze
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of extracted keywords
        """
        # This is a simple keyword extraction implementation
        # Could be enhanced with NLP libraries in a real implementation
        
        # Get focus area keywords from config
        focus_area_keywords = []
        focus_areas = self.config.get("user_options", {}).get("focus_areas", {})
        for area, info in focus_areas.items():
            if "keywords" in info:
                focus_area_keywords.extend(info["keywords"])
        
        # Create a basic keyword frequency counter
        import re
        from collections import Counter
        
        # Tokenize and clean text
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stopwords (simplified list)
        stopwords = {"the", "and", "to", "of", "a", "in", "that", "it", "with", "for", "on", "is", "was", "be", "this", "are"}
        filtered_words = [word for word in words if word not in stopwords]
        
        # Count frequencies
        word_counts = Counter(filtered_words)
        
        # Prioritize focus area keywords that appear in the text
        prioritized_keywords = [word for word in focus_area_keywords if word in word_counts]
        
        # Add other frequent words up to max_keywords
        remaining_slots = max_keywords - len(prioritized_keywords)
        if remaining_slots > 0:
            other_keywords = [word for word, _ in word_counts.most_common(remaining_slots) 
                             if word not in prioritized_keywords]
            prioritized_keywords.extend(other_keywords)
        
        return prioritized_keywords[:max_keywords]