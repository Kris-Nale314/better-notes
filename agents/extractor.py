# agents/extractor.py
from typing import Dict, Any, List, Optional
from .base import BaseAgent

class ExtractorAgent(BaseAgent):
    """
    Agent specialized in extracting specific information from document chunks.
    """
    
    def __init__(
        self,
        llm_client,
        crew_type: str,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        max_chunk_size: int = 1500  # Add this parameter
    ):
        """
        Initialize an extractor agent.
        
        Args:
            llm_client: LLM client for agent communication
            crew_type: Type of crew (issues, actions, opportunities)
            config: Optional pre-loaded configuration
            verbose: Whether to enable verbose mode
            max_chunk_size: Maximum size of text chunks to process
        """
        super().__init__(
            llm_client=llm_client,
            agent_type="extraction",
            crew_type=crew_type,
            config=config,
            verbose=verbose,
            max_chunk_size=max_chunk_size  # Pass this parameter to BaseAgent
        )
    
    def extract_from_chunk(self, chunk: str, document_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract information from a document chunk.
        
        Args:
            chunk: Text chunk to analyze
            document_info: Optional document metadata
            
        Returns:
            Extraction results
        """
        # Ensure chunk size is within limits
        safe_chunk = self.truncate_text(chunk)
        
        # Prepare context for prompt building
        context = {
            "chunk_text": safe_chunk,
            "document_info": document_info or {}
        }
        
        # Execute the extraction task
        return self.execute_task(context=context)
    
    def extract_from_chunk(self, chunk: str, document_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract information from a document chunk.
        
        Args:
            chunk: Text chunk to analyze
            document_info: Optional document metadata
            
        Returns:
            Extraction results
        """
        # Prepare context for prompt building
        context = {
            "chunk_text": chunk,
            "document_info": document_info or {}
        }
        
        # Execute the extraction task
        return self.execute_task(context=context)