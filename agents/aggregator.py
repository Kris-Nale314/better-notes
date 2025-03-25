# agents/aggregator.py
"""
Aggregator Agent - Specialized in combining and deduplicating extraction results.
Works with ProcessingContext and integrated crew architecture.
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
    Enhances metadata during the aggregation process and works with ProcessingContext.
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
            agent_type="aggregation",
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
        
        # Aggregate the results
        aggregated_result = self.aggregate_results(
            extraction_results=extraction_results, 
            document_info=context.document_info
        )
        
        return aggregated_result
    
    # In agents/aggregator.py:
    async def aggregate_results(
        self, 
        extraction_results: List[Dict[str, Any]], 
        document_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Aggregate and deduplicate results from multiple extractions.
        Enhances metadata during aggregation process.
        
        Args:
            extraction_results: List of extraction results from chunks
            document_info: Optional document metadata
            
        Returns:
            Aggregated results with deduplication and enhanced metadata
        """
        # Prepare context for prompt building
        context = {
            "extraction_results": self._preprocess_extraction_results(extraction_results),
            "document_info": document_info or {},
            "result_count": len(extraction_results)
        }
        
        # Add statistics for better context
        successful_chunks = sum(1 for r in extraction_results if not (isinstance(r, dict) and "error" in r))
        context["success_rate"] = f"{successful_chunks}/{len(extraction_results)} chunks processed successfully"
        
        # Calculate total items across all extractions
        item_count = 0
        key_field = self.get_key_field()
        for result in extraction_results:
            if isinstance(result, dict) and key_field in result:
                items = result[key_field]
                if isinstance(items, list):
                    item_count += len(items)
        
        context["total_items"] = item_count
        
        # Execute the aggregation task - ADD AWAIT HERE
        result = await self.execute_task(context=context)
        
        # Enhance the result with aggregation metadata
        result = self._enhance_result_with_metadata(result, extraction_results)
        
        return result
    
    def _get_stage_specific_content(self, context) -> str:
        """Get stage-specific content for the prompt."""
        # If context is a dictionary with extraction_results
        if isinstance(context, dict) and "extraction_results" in context:
            extraction_results = context["extraction_results"]
            result_count = context.get("result_count", len(extraction_results))
            success_rate = context.get("success_rate", "")
            total_items = context.get("total_items", 0)
            
            # Format extraction results
            extraction_summary = json.dumps(extraction_results, indent=2, default=str)
            
            # Add truncation if too long
            if len(extraction_summary) > 3000:
                extraction_summary = extraction_summary[:3000] + "...\n[Output truncated]"
            
            return f"""
            EXTRACTION RESULTS:
            {extraction_summary}
            
            EXTRACTION STATISTICS:
            - Total chunks processed: {result_count}
            - Success rate: {success_rate}
            - Total items extracted: {total_items}
            """
        
        # Otherwise, return empty string
        return ""
    
    def _preprocess_extraction_results(self, extraction_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess extraction results to normalize format and extract key information.
        
        Args:
            extraction_results: Raw extraction results
            
        Returns:
            Preprocessed results
        """
        processed_results = []
        key_field = self.get_key_field()
        
        for result in extraction_results:
            # Skip error results
            if isinstance(result, dict) and "error" in result:
                continue
                
            # Get the items
            items = []
            if isinstance(result, dict) and key_field in result:
                items = result[key_field]
                if not isinstance(items, list):
                    items = []
            
            # Add to processed results
            processed_results.append({
                key_field: items,
                "chunk_index": result.get("_metadata", {}).get("chunk_index", 0) if isinstance(result, dict) else 0,
                "count": len(items)
            })
        
        return processed_results
    
    def _enhance_result_with_metadata(self, result: Any, extraction_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enhance aggregation result with additional metadata.
        
        Args:
            result: Aggregation result
            extraction_results: Original extraction results
            
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
                key_field = self.get_key_field(aggregate=True)
                result = {
                    key_field: [{"description": result}]
                }
        
        # Handle non-dictionary results
        if not isinstance(result, dict):
            key_field = self.get_key_field(aggregate=True)
            result = {
                key_field: [{"description": str(result)}]
            }
        
        # Ensure the key field exists
        key_field = self.get_key_field(aggregate=True)
        if key_field not in result:
            result[key_field] = []
        
        # Add aggregation metadata
        key_field = self.get_key_field(aggregate=True)
        items = result[key_field]
        
        if isinstance(items, list):
            # Calculate source chunks for each item
            source_chunks = {}
            extracted_key_field = self.get_key_field()
            
            for ext_result in extraction_results:
                if not isinstance(ext_result, dict) or extracted_key_field not in ext_result:
                    continue
                    
                chunk_index = ext_result.get("_metadata", {}).get("chunk_index", 0)
                ext_items = ext_result[extracted_key_field]
                
                if not isinstance(ext_items, list):
                    continue
                    
                # Record chunk index for each item
                for ext_item in ext_items:
                    if not isinstance(ext_item, dict):
                        continue
                        
                    item_title = ext_item.get("title", "")
                    item_desc = ext_item.get("description", "")
                    
                    # Create a simple key for matching
                    item_key = (item_title + " " + item_desc).lower()[:100]
                    
                    if item_key not in source_chunks:
                        source_chunks[item_key] = []
                        
                    source_chunks[item_key].append(chunk_index)
            
            # Enhance each aggregated item
            for item in items:
                if not isinstance(item, dict):
                    continue
                    
                # Generate a key for this item
                item_title = item.get("title", "")
                item_desc = item.get("description", "")
                item_key = (item_title + " " + item_desc).lower()[:100]
                
                # Add mention count if not present
                if "mention_count" not in item and item_key in source_chunks:
                    item["mention_count"] = len(source_chunks[item_key])
                
                # Add source chunks if not present
                if "source_chunks" not in item and item_key in source_chunks:
                    item["source_chunks"] = source_chunks[item_key]
                
                # Add confidence score if not present
                if "confidence" not in item and "mention_count" in item:
                    # Simple confidence heuristic based on mention count
                    if item["mention_count"] >= 3:
                        item["confidence"] = "high"
                    elif item["mention_count"] == 2:
                        item["confidence"] = "medium"
                    else:
                        item["confidence"] = "low"
                
                # Ensure keywords are present
                if "keywords" not in item and "description" in item:
                    try:
                        item["keywords"] = self.extract_keywords(item["description"])
                    except Exception as e:
                        logger.warning(f"Error extracting keywords: {e}")
                        item["keywords"] = []
        
        # Add overall metadata
        result["_metadata"] = {
            "aggregated_count": len(items) if isinstance(items, list) else 0,
            "original_count": sum(len(r.get(self.get_key_field(), [])) for r in extraction_results if isinstance(r, dict)),
            "processed_chunks": len(extraction_results),
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def get_key_field(self, aggregate: bool = False) -> str:
        """
        Get the key field name for the items being aggregated based on crew type.
        
        Args:
            aggregate: Whether to get the aggregated field name
            
        Returns:
            Field name for items (e.g., "issues", "action_items", "opportunities")
        """
        # Default mapping
        field_mapping = {
            "issues": "issues",
            "actions": "action_items",
            "opportunities": "opportunities",
            "risks": "risks"
        }
        
        # Get from config if available
        key_field = field_mapping.get(self.crew_type, f"{self.crew_type}_items")
        
        # If aggregated field, add prefix
        if aggregate:
            return f"aggregated_{key_field}"
        
        return key_field