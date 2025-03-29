"""
Simple Aggregator Agent for Better Notes that deduplicates and combines extraction results.
Focuses on letting the LLM do the heavy lifting for combining similar items.
"""

import json
import logging
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base import BaseAgent

logger = logging.getLogger(__name__)

class AggregatorAgent(BaseAgent):
    """
    Simplified Aggregator agent that combines and deduplicates extraction results.
    Lets the LLM handle most of the aggregation logic for reliable results.
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
        
        logger.info(f"AggregatorAgent initialized for {crew_type}")
    
    async def process(self, context):
        """
        Process extraction results using the context.
        
        Args:
            context: ProcessingContext object
            
        Returns:
            Aggregated results
        """
        logger.info("AggregatorAgent starting aggregation process")
        
        try:
            # Get extraction results from context
            extraction_results = context.results.get("extraction", [])
            
            if not extraction_results:
                logger.warning("No extraction results found in context for aggregation")
                return self._create_empty_result()
            
            # Aggregate results
            aggregated_result = await self.aggregate_results(
                extraction_results=extraction_results,
                document_info=getattr(context, 'document_info', {})
            )
            
            logger.info(f"Successfully aggregated results from {len(extraction_results)} extraction results")
            return aggregated_result
            
        except Exception as e:
            logger.error(f"Error in aggregation process: {e}")
            logger.error(traceback.format_exc())
            
            # Return empty result in case of error
            return self._create_empty_result()
    
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
        
        # Execute aggregation with the LLM doing most of the work
        try:
            result = await self.execute_task(aggregation_context)
            
            # Enhance result with metadata
            enhanced_result = self._enhance_result_with_metadata(result, extraction_results)
            
            # Validate result structure
            validated_result = self._validate_aggregation_result(enhanced_result)
            
            return validated_result
            
        except Exception as e:
            logger.error(f"Error in aggregate_results: {e}")
            logger.error(traceback.format_exc())
            
            # Create a simple aggregation of items in case of error
            return self._create_fallback_aggregation(extraction_results)
    
    def _get_stage_specific_content(self, context) -> str:
        """
        Get stage-specific content for the prompt with focus on effective aggregation.
        
        Args:
            context: Aggregation context
            
        Returns:
            Stage-specific content string
        """
        if isinstance(context, dict) and "extraction_results" in context:
            # Add statistics about extraction
            content = f"""
            EXTRACTION STATISTICS:
            - Total chunks processed: {context.get('total_chunks', 0)}
            - Total items found: {context.get('items_found', 0)}
            
            AGGREGATION TASK:
            1. Combine similar items from different chunks into single comprehensive items
            2. Remove exact duplicates
            3. Preserve important variations and details when merging similar items
            4. Keep unique items even if they only appear in one chunk
            5. If an item appears in multiple chunks, combine the contexts from each occurrence
            
            Your primary goal is to eliminate redundancy while preserving all unique information.
            """
            
            # Add extraction results in a formatted way
            results = context.get("extraction_results", [])
            
            # Convert results to a more LLM-friendly format
            simplified_results = self._simplify_results_for_llm(results)
            
            # Add simplified results to the prompt
            content += "\n\nEXTRACTION RESULTS:\n"
            content += simplified_results
            
            # Add specific output format guidance
            content += f"\n\nOUTPUT FORMAT:\nReturn the aggregated {self.crew_type} as a JSON array under the key '{self._get_output_field_name()}'."
            content += "\nEach item should preserve all relevant fields from the input items, combining similar information."
            
            if self.crew_type == "issues":
                content += "\n\nFor issues specifically, combine similar issues into a single comprehensive issue with:"
                content += "\n- A clear title that captures the essence of the issue"
                content += "\n- A detailed description that combines information from all mentions"
                content += "\n- The highest severity level from any of the merged issues"
                content += "\n- The most appropriate category based on the combined information"
                content += "\n- Source information showing which chunks the issue appeared in"
            
            return content
            
        return ""
    
    def _simplify_results_for_llm(self, results: List[Dict[str, Any]]) -> str:
        """
        Simplify extraction results into a more concise format for the LLM.
        
        Args:
            results: Extraction results
            
        Returns:
            Simplified results as a string
        """
        items_field = self._get_input_field_name()
        all_items = []
        
        # Collect all items from all chunks with chunk identifiers
        for i, result in enumerate(results):
            if not isinstance(result, dict):
                continue
                
            # Skip error results
            if "error" in result and not items_field in result:
                continue
                
            # Get items from this chunk
            chunk_items = result.get(items_field, [])
            if not chunk_items:
                continue
                
            # Get chunk metadata
            chunk_index = result.get("_metadata", {}).get("chunk_index", i)
            chunk_position = result.get("_metadata", {}).get("position", "unknown")
            
            # Add each item with chunk identifier
            for item in chunk_items:
                if not isinstance(item, dict):
                    continue
                    
                # Add chunk information to the item
                item_with_chunk = {
                    "chunk_index": chunk_index,
                    "chunk_position": chunk_position
                }
                
                # Copy relevant fields
                for field in ["title", "description", "severity", "category"]:
                    if field in item:
                        item_with_chunk[field] = item[field]
                
                all_items.append(item_with_chunk)
        
        # Organize items in a format that's easy for the LLM to process
        if len(all_items) <= 15:
            # For a small number of items, include everything in full detail
            return json.dumps(all_items, indent=2)
        else:
            # For a larger number of items, create a more compact format
            compact_items = []
            
            for item in all_items:
                compact_item = {
                    "chunk": f"{item.get('chunk_index', 0)} ({item.get('chunk_position', 'unknown')})",
                    "title": item.get("title", "Untitled"),
                }
                
                # Add other fields if they're not too long
                desc = item.get("description", "")
                if len(desc) > 100:
                    desc = desc[:97] + "..."
                compact_item["desc"] = desc
                
                if "severity" in item:
                    compact_item["sev"] = item["severity"]
                    
                if "category" in item:
                    compact_item["cat"] = item["category"]
                
                compact_items.append(compact_item)
            
            return json.dumps(compact_items, indent=1)
    
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
            if isinstance(result, dict) and "error" in result and not items_field in result:
                continue
            
            # Extract items from the result
            if isinstance(result, dict) and items_field in result:
                items = result[items_field]
                if isinstance(items, list) and items:
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
        # Parse the result if it's a string
        if isinstance(result, str):
            try:
                # Try to parse as JSON
                parsed_result = self.parse_llm_json(result)
                if isinstance(parsed_result, dict):
                    result = parsed_result
                else:
                    # Create basic structure
                    result = {self._get_output_field_name(): []}
            except Exception as e:
                logger.error(f"Error parsing result: {e}")
                # Not valid JSON, create basic structure
                result = {self._get_output_field_name(): []}
        
        # If result is still not a dictionary, create basic structure
        if not isinstance(result, dict):
            result = {self._get_output_field_name(): []}
        
        # Ensure output field exists
        output_field = self._get_output_field_name()
        if output_field not in result:
            result[output_field] = []
        
        # Add source chunks and mention counts if not present
        items = result[output_field]
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    # Find source chunks if not already present
                    if "source_chunks" not in item:
                        item["source_chunks"] = self._find_source_chunks(item, extraction_results)
                    
                    # Add mention count if not present
                    if "mention_count" not in item and "source_chunks" in item:
                        item["mention_count"] = len(item["source_chunks"])
                    
                    # Add confidence based on mention count
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
    
    def _validate_aggregation_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the structure of the aggregation result and fix any issues.
        
        Args:
            result: The aggregation result to validate
            
        Returns:
            Validated result
        """
        output_field = self._get_output_field_name()
        
        # Ensure result is a dictionary
        if not isinstance(result, dict):
            logger.warning(f"Aggregation result is not a dictionary: {type(result)}")
            return {
                output_field: [],
                "_metadata": {
                    "input_count": 0,
                    "output_count": 0,
                    "deduplication_rate": 0,
                    "chunks_processed": 0,
                    "error": "Invalid result format"
                }
            }
        
        # Ensure output field exists and is a list
        if output_field not in result or not isinstance(result[output_field], list):
            logger.warning(f"Output field missing or not a list: {output_field}")
            result[output_field] = []
        
        # Validate each item
        valid_items = []
        for item in result[output_field]:
            if not isinstance(item, dict):
                logger.warning(f"Item is not a dictionary: {item}")
                continue
                
            # For issues analysis, ensure minimum required fields
            if self.crew_type == "issues":
                # Ensure title exists
                if "title" not in item or not item["title"]:
                    if "description" in item:
                        # Generate title from description
                        item["title"] = self._generate_title(item["description"])
                    else:
                        logger.warning(f"Item missing both title and description")
                        continue
                
                # Ensure description exists
                if "description" not in item or not item["description"]:
                    if "title" in item:
                        item["description"] = item["title"]
                    else:
                        logger.warning(f"Item missing both title and description")
                        continue
            
            valid_items.append(item)
        
        # Replace items with validated list
        result[output_field] = valid_items
        
        # Update item count in metadata
        if "_metadata" in result:
            result["_metadata"]["output_count"] = len(valid_items)
        
        return result
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """
        Create an empty result when no extraction results are available.
        
        Returns:
            Empty result structure
        """
        return {
            self._get_output_field_name(): [],
            "_metadata": {
                "input_count": 0,
                "output_count": 0,
                "deduplication_rate": 0,
                "chunks_processed": 0,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _create_fallback_aggregation(self, extraction_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a simple aggregation when LLM-based aggregation fails.
        
        Args:
            extraction_results: List of extraction results
            
        Returns:
            Simple aggregated result
        """
        logger.info("Creating fallback aggregation")
        
        # Initialize result
        result = {
            self._get_output_field_name(): [],
            "_metadata": {
                "input_count": 0,
                "output_count": 0,
                "deduplication_rate": 0,
                "chunks_processed": len(extraction_results),
                "fallback": True,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Get all items
        input_field = self._get_input_field_name()
        all_items = []
        title_map = {}  # For simple deduplication by title
        
        for i, extraction in enumerate(extraction_results):
            if not isinstance(extraction, dict) or input_field not in extraction:
                continue
                
            items = extraction.get(input_field, [])
            if not isinstance(items, list):
                continue
                
            chunk_index = extraction.get("_metadata", {}).get("chunk_index", i)
            
            for item in items:
                if not isinstance(item, dict):
                    continue
                    
                # Skip items without title or description
                if ("title" not in item or not item["title"]) and ("description" not in item or not item["description"]):
                    continue
                
                # Generate title if missing
                if "title" not in item or not item["title"]:
                    if "description" in item:
                        item["title"] = self._generate_title(item["description"])
                    else:
                        continue
                
                # Simple deduplication by title
                title = item["title"].lower()
                if title in title_map:
                    # We already have an item with this title, update source chunks
                    existing_item = title_map[title]
                    if "source_chunks" not in existing_item:
                        existing_item["source_chunks"] = [chunk_index]
                    elif chunk_index not in existing_item["source_chunks"]:
                        existing_item["source_chunks"].append(chunk_index)
                else:
                    # New item, add to our collection
                    item_copy = item.copy()
                    item_copy["source_chunks"] = [chunk_index]
                    all_items.append(item_copy)
                    title_map[title] = item_copy
        
        # Update result
        result[self._get_output_field_name()] = all_items
        result["_metadata"]["input_count"] = self._count_total_items(extraction_results)
        result["_metadata"]["output_count"] = len(all_items)
        
        # Calculate deduplication rate
        input_count = result["_metadata"]["input_count"]
        output_count = result["_metadata"]["output_count"]
        if input_count > 0:
            result["_metadata"]["deduplication_rate"] = round((input_count - output_count) / input_count * 100)
        
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
        input_field = self._get_input_field_name()
        
        for result in extraction_results:
            if not isinstance(result, dict) or input_field not in result:
                continue
                
            chunk_index = result.get("_metadata", {}).get("chunk_index", -1)
            if chunk_index == -1:
                continue
                
            extracted_items = result.get(input_field, [])
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
    
    def _have_significant_overlap(self, text1: str, text2: str, threshold: float = 0.3) -> bool:
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
            "risks": "risks",
            "insights": "insights"
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
    
    def _generate_title(self, description: str, max_length: int = 50) -> str:
        """
        Generate a title from a description if one wasn't provided.
        
        Args:
            description: Item description
            max_length: Maximum title length
            
        Returns:
            Generated title
        """
        if not description:
            return "Untitled Item"
            
        # Try to extract the first sentence
        import re
        first_sentence_match = re.match(r'^([^.!?]+[.!?])', description)
        
        if first_sentence_match:
            title = first_sentence_match.group(1).strip()
        else:
            # Just use the first part of the description
            title = description.strip()
        
        # Truncate if needed
        if len(title) > max_length:
            title = title[:max_length].strip() + "..."
            
        return title