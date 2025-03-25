"""
Aggregator Agent - Specialized in combining and deduplicating extraction results.
Clean implementation that leverages the new BaseAgent architecture.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base import BaseAgent

logger = logging.getLogger(__name__)

class AggregatorAgent(BaseAgent):
    """
    Agent specialized in combining and deduplicating extraction results from multiple chunks.
    Enhances metadata during the aggregation process.
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
        """Initialize an aggregator agent."""
        super().__init__(
            llm_client=llm_client,
            agent_type="aggregator",
            crew_type=crew_type,
            config=config,
            config_manager=config_manager,
            verbose=verbose,
            max_chunk_size=max_chunk_size,
            max_rpm=max_rpm
        )
    
    async def process(self, context):
        """
        Process extraction results using the context.
        
        Args:
            context: ProcessingContext object
            
        Returns:
            Aggregated results
        """
        # Get extraction results from context
        extraction_results = context.results.get("extraction", [])
        
        # Aggregate results
        aggregated_result = await self.aggregate_results(
            extraction_results=extraction_results,
            document_info=context.document_info
        )
        
        return aggregated_result
    
    async def aggregate_results(
        self, 
        extraction_results: List[Dict[str, Any]], 
        document_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Aggregate and deduplicate results from multiple extractions.
        
        Args:
            extraction_results: List of extraction results from chunks
            document_info: Optional document metadata
            
        Returns:
            Aggregated results with deduplication and enhanced metadata
        """
        # Prepare aggregation context
        aggregation_context = {
            "extraction_results": self._preprocess_extraction_results(extraction_results),
            "document_info": document_info or {},
            "total_chunks": len(extraction_results),
            "items_found": self._count_total_items(extraction_results)
        }
        
        # Execute aggregation
        result = await self.execute_task(aggregation_context)
        
        # Enhance result with metadata
        return self._enhance_result_with_metadata(result, extraction_results)
    
    def _get_stage_specific_content(self, context) -> str:
        """Get stage-specific content for the prompt."""
        if isinstance(context, dict) and "extraction_results" in context:
            # Add statistics about extraction
            content = f"""
            EXTRACTION STATISTICS:
            - Total chunks processed: {context.get('total_chunks', 0)}
            - Total items found: {context.get('items_found', 0)}
            
            EXTRACTION RESULTS:
            """
            
            # Add extraction results in a formatted way
            results = context.get("extraction_results", [])
            
            # Limit the detail to avoid token overload
            results_summary = json.dumps(results, indent=2)
            if len(results_summary) > 3000:
                # Truncate and add indication
                results_summary = results_summary[:3000] + "\n...(truncated for brevity)..."
            
            content += results_summary
            
            return content
            
        return ""
    
    def _preprocess_extraction_results(self, extraction_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess extraction results for easier aggregation.
        
        Args:
            extraction_results: Raw extraction results
            
        Returns:
            Preprocessed results focusing on items only
        """
        preprocessed = []
        items_field = self._get_input_field_name()
        
        for i, result in enumerate(extraction_results):
            # Skip error results
            if isinstance(result, dict) and "error" in result:
                continue
            
            # Extract items from the result
            if isinstance(result, dict) and items_field in result:
                items = result[items_field]
                if isinstance(items, list):
                    chunk_info = {
                        "chunk_index": result.get("_metadata", {}).get("chunk_index", i),
                        "position": result.get("_metadata", {}).get("position", "unknown"),
                        "items": items
                    }
                    preprocessed.append(chunk_info)
        
        return preprocessed
    
    def _count_total_items(self, extraction_results: List[Dict[str, Any]]) -> int:
        """
        Count the total number of items across all extraction results.
        
        Args:
            extraction_results: List of extraction results
            
        Returns:
            Total item count
        """
        count = 0
        items_field = self._get_input_field_name()
        
        for result in extraction_results:
            if isinstance(result, dict) and items_field in result:
                items = result[items_field]
                if isinstance(items, list):
                    count += len(items)
        
        return count
    
    def _enhance_result_with_metadata(self, result: Any, extraction_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enhance aggregation result with additional metadata.
        
        Args:
            result: Aggregation result
            extraction_results: Original extraction results
            
        Returns:
            Enhanced result with metadata
        """
        # Ensure result is a dictionary
        if isinstance(result, str):
            try:
                # Try to parse as JSON
                parsed_result = self.parse_llm_json(result)
                if isinstance(parsed_result, dict):
                    result = parsed_result
                else:
                    # Create basic structure
                    result = {self._get_output_field_name(): [{"description": result}]}
            except:
                # Not valid JSON, create basic structure
                result = {self._get_output_field_name(): [{"description": result}]}
        
        # Handle non-dictionary results
        if not isinstance(result, dict):
            result = {self._get_output_field_name(): [{"description": str(result)}]}
        
        # Ensure output field exists
        output_field = self._get_output_field_name()
        if output_field not in result:
            result[output_field] = []
        
        # Add mention counts and source chunks
        items = result[output_field]
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    # Calculate source info if not present
                    if "source_chunks" not in item:
                        item["source_chunks"] = self._find_source_chunks(item, extraction_results)
                    
                    # Add mention count if not present
                    if "mention_count" not in item and "source_chunks" in item:
                        item["mention_count"] = len(item["source_chunks"])
                    
                    # Add confidence score if not present
                    if "confidence" not in item and "mention_count" in item:
                        item["confidence"] = self._calculate_confidence(item["mention_count"])
        
        # Add aggregation metadata
        input_count = self._count_total_items(extraction_results)
        output_count = len(items) if isinstance(items, list) else 0
        
        result["_metadata"] = {
            "input_count": input_count,
            "output_count": output_count,
            "deduplication_rate": round((input_count - output_count) / input_count * 100) if input_count else 0,
            "chunks_processed": len(extraction_results),
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def _find_source_chunks(self, item: Dict[str, Any], extraction_results: List[Dict[str, Any]]) -> List[int]:
        """
        Find source chunks for an aggregated item.
        
        Args:
            item: Aggregated item
            extraction_results: Original extraction results
            
        Returns:
            List of source chunk indices
        """
        # Use item title or description as matching key
        title = item.get("title", "").lower()
        description = item.get("description", "").lower()
        
        source_chunks = []
        items_field = self._get_input_field_name()
        
        for result in extraction_results:
            if not isinstance(result, dict) or items_field not in result:
                continue
                
            chunk_index = result.get("_metadata", {}).get("chunk_index", -1)
            if chunk_index == -1:
                continue
                
            extracted_items = result[items_field]
            if not isinstance(extracted_items, list):
                continue
            
            # Check for matches
            for extracted_item in extracted_items:
                if not isinstance(extracted_item, dict):
                    continue
                    
                ext_title = extracted_item.get("title", "").lower()
                ext_desc = extracted_item.get("description", "").lower()
                
                # Determine if this is likely the same item
                if (title and ext_title and (title in ext_title or ext_title in title)) or \
                   (description and ext_desc and (
                       description in ext_desc or 
                       ext_desc in description or
                       self._have_significant_overlap(description, ext_desc)
                   )):
                    if chunk_index not in source_chunks:
                        source_chunks.append(chunk_index)
        
        return source_chunks
    
    def _have_significant_overlap(self, text1: str, text2: str, threshold: float = 0.5) -> bool:
        """
        Check if two texts have significant word overlap.
        
        Args:
            text1: First text
            text2: Second text
            threshold: Overlap threshold (0.0 to 1.0)
            
        Returns:
            True if significant overlap exists
        """
        # Simple word-based overlap calculation
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return (intersection / union) >= threshold if union > 0 else False
    
    def _calculate_confidence(self, mention_count: int) -> str:
        """
        Calculate confidence level based on mention count.
        
        Args:
            mention_count: Number of mentions
            
        Returns:
            Confidence level string
        """
        if mention_count >= 3:
            return "high"
        elif mention_count == 2:
            return "medium"
        else:
            return "low"
    
    def _get_input_field_name(self) -> str:
        """
        Get the field name for input items.
        
        Returns:
            Field name for input items
        """
        # Map crew types to input field names
        field_map = {
            "issues": "issues",
            "actions": "action_items",
            "opportunities": "opportunities",
            "risks": "risks"
        }
        
        return field_map.get(self.crew_type, f"{self.crew_type}_items")
    
    def _get_output_field_name(self) -> str:
        """
        Get the field name for aggregated output items.
        
        Returns:
            Field name for output items
        """
        input_field = self._get_input_field_name()
        return f"aggregated_{input_field}"