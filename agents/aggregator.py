# agents/aggregator.py
from typing import Dict, Any, List, Optional
from .base import BaseAgent

class AggregatorAgent(BaseAgent):
    """
    Agent specialized in combining and deduplicating extraction results from multiple chunks.
    """
    
    def __init__(
        self,
        llm_client,
        crew_type: str,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        max_chunk_size: int = 1500, 
        max_rpm: int = 10  # Add this parameter
    ):
        """
        Initialize an aggregator agent.
        
        Args:
            llm_client: LLM client for agent communication
            crew_type: Type of crew (issues, actions, opportunities)
            config: Optional pre-loaded configuration
            verbose: Whether to enable verbose mode
            max_chunk_size: Maximum size of text chunks to process
        """
        super().__init__(
            llm_client=llm_client,
            agent_type="aggregation",
            crew_type=crew_type,
            config=config,
            verbose=verbose,
            max_chunk_size=max_chunk_size, 
            max_rpm = max_rpm  # Pass this parameter to BaseAgent
        )
    
    # Rest of the implementation...
    
    def aggregate_results(self, extraction_results: List[Dict[str, Any]], document_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Aggregate and deduplicate results from multiple extractions.
        
        Args:
            extraction_results: List of extraction results from chunks
            document_info: Optional document metadata
            
        Returns:
            Aggregated results with deduplication and mention tracking
        """
        # Prepare context for prompt building
        context = {
            "extraction_results": extraction_results,
            "document_info": document_info or {},
            "result_count": len(extraction_results)
        }
        
        # Execute the aggregation task
        return self.execute_task(context=context)

    def get_key_field(self) -> str:
        """
        Get the key field name for the items being aggregated based on crew type.
        
        Returns:
            Field name for items (e.g., "issues", "action_items", "opportunities")
        """
        key_field_map = {
            "issues": "issues",
            "actions": "action_items",
            "opportunities": "opportunities"
        }
        return key_field_map.get(self.crew_type, "items")
    
    def pre_process_results(self, extraction_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optional pre-processing of extraction results before aggregation.
        This can be used to standardize formats, extract specific fields, etc.
        
        Args:
            extraction_results: Raw extraction results
            
        Returns:
            Pre-processed results ready for aggregation
        """
        # Get the key field name for this crew type
        key_field = self.get_key_field()
        
        # Pre-process and standardize the results
        processed_results = []
        for i, result in enumerate(extraction_results):
            # If the result is a string (e.g., from direct agent output), try to parse it
            if isinstance(result, str):
                try:
                    import json
                    parsed = json.loads(result)
                    if isinstance(parsed, dict):
                        result = parsed
                except:
                    # If parsing fails, wrap the string in a simple dict
                    result = {"text": result, "chunk_index": i}
            
            # Ensure the result is a dictionary
            if not isinstance(result, dict):
                result = {"data": result, "chunk_index": i}
            
            # Add chunk index if not present
            if "chunk_index" not in result:
                result["chunk_index"] = i
                
            processed_results.append(result)
            
        return processed_results