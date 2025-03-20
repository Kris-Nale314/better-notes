"""
Summarizer module that processes individual chunks with metadata extraction.
"""

import json
import re
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class ChunkSummarizer:
    """Summarizes document chunks and extracts metadata."""
    
    def __init__(self, llm_client):
        """
        Initialize the chunk summarizer.
        
        Args:
            llm_client: LLM client for generating summaries
        """
        self.llm_client = llm_client
    
    async def summarize_chunk(self, 
                            chunk: Dict[str, Any], 
                            document_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Summarize a single chunk with metadata extraction.
        
        Args:
            chunk: Chunk dictionary with text and metadata
            document_context: Optional context from document analysis
            
        Returns:
            Dictionary with summary and extracted metadata
        """
        # Extract chunk info
        text = chunk['text']
        chunk_index = chunk.get('index', 0)
        chunk_type = chunk.get('chunk_type', 'unknown')
        
        # Determine chunk position for context
        position = self._determine_position(chunk_index, chunk_type, document_context)
        
        # Create prompt based on document context and chunk position
        prompt = self._create_summary_prompt(text, position, document_context)
        
        # Get summary from LLM
        try:
            response = await self.llm_client.generate_completion_async(prompt)
            
            # Parse JSON response
            try:
                result = json.loads(response)
                
                # Add chunk metadata
                result.update({
                    'chunk_index': chunk_index,
                    'position': position,
                    'chunk_type': chunk_type
                })
                
                return result
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                logger.warning(f"Failed to parse JSON from chunk {chunk_index}. Using fallback extraction.")
                return self._fallback_extraction(response, chunk_index, position, chunk_type)
        except Exception as e:
            logger.error(f"Error in chunk {chunk_index} summarization: {e}")
            raise
    
    def _determine_position(self, 
                         chunk_index: int, 
                         chunk_type: str,
                         document_context: Optional[Dict[str, Any]]) -> str:
        """
        Determine the position/role of a chunk in the document.
        
        Args:
            chunk_index: Index of the chunk
            chunk_type: Type of the chunk
            document_context: Optional document context
            
        Returns:
            Position string ('introduction', 'body', 'conclusion', etc.)
        """
        # Get total chunks if available
        total_chunks = document_context.get('total_chunks', 3) if document_context else 3
        
        # Simple position determination based on index
        if chunk_index == 0:
            return 'introduction'
        elif chunk_index == total_chunks - 1:
            return 'conclusion'
        elif chunk_index < total_chunks * 0.3:
            return 'early'
        elif chunk_index > total_chunks * 0.7:
            return 'late'
        else:
            return 'middle'
    
    def _create_summary_prompt(self, 
                             text: str, 
                             position: str, 
                             document_context: Optional[Dict[str, Any]]) -> str:
        """
        Create a summary prompt tailored to the chunk's position and document type.
        
        Args:
            text: Chunk text
            position: Chunk position
            document_context: Optional document context
            
        Returns:
            Prompt string
        """
        # Default context if none provided
        if not document_context:
            document_context = {}
        
        # Extract context information
        is_transcript = document_context.get('is_meeting_transcript', False)
        client = document_context.get('client_name', None)
        domain_categories = document_context.get('domain_categories', [])
        key_topics = document_context.get('key_topics', [])
        
        # Create position-specific instructions
        position_instructions = {
            'introduction': """
                This is the INTRODUCTION of the document.
                Focus on identifying the purpose, participants, and main agenda items.
                Capture the context-setting information that frames the rest of the document.
            """,
            'conclusion': """
                This is the CONCLUSION of the document.
                Focus on final decisions, next steps, action items, and closing remarks.
                Capture any deadlines, commitments, or follow-up plans.
            """,
            'early': """
                This is an EARLY section of the document.
                Focus on initial topics, background information, and preliminary discussions.
                Capture information that helps establish the purpose and direction.
            """,
            'middle': """
                This is a MIDDLE section of the document.
                Focus on the main discussions, key points, and ongoing topics.
                Capture substantive content and important exchanges.
            """,
            'late': """
                This is a LATE section of the document, approaching the conclusion.
                Focus on resolutions, emerging consensus, and preparation for wrap-up.
                Capture decisions that are being finalized and solidified.
            """
        }
        
        # Create type-specific instructions
        type_instructions = ""
        if is_transcript:
            type_instructions = """
                This appears to be a meeting transcript or conversation.
                Preserve speaker attributions for important statements.
                Capture the dynamics and flow of the conversation.
                Note areas of agreement, disagreement, and question-answer exchanges.
            """
            
            # Add client context if available
            if client:
                type_instructions += f"\nThis appears to be related to client: {client}."
                
        # Build the complete prompt
        prompt = f"""
        Summarize this document section and extract key metadata.

        {position_instructions.get(position, '')}
        {type_instructions}
        
        {f'Relevant business domains: {", ".join(domain_categories)}' if domain_categories else ''}
        {f'Key topics to watch for: {", ".join(key_topics)}' if key_topics else ''}

        Return the result in JSON format with these fields:
        - summary: A detailed summary of this section that captures key points, discussions, and decisions
        - keywords: A list of 5-10 key topics/terms that best represent this section
        - speakers: List of speakers in this section (if identifiable)
        - importance: A score from 1-5 indicating how important this section is (5 being highest)
        - action_items: List of any action items, tasks, commitments mentioned in this section
        - key_quotes: Any particularly important or insightful quotes from this section (with attribution if possible)
        
        SECTION TEXT:
        {text}
        """
        
        return prompt
    
    def _fallback_extraction(self, 
                           text_result: str, 
                           chunk_index: int, 
                           position: str, 
                           chunk_type: str) -> Dict[str, Any]:
        """
        Extract summary and metadata when JSON parsing fails.
        
        Args:
            text_result: Text result from LLM
            chunk_index: Index of the chunk
            position: Position string
            chunk_type: Type of the chunk
            
        Returns:
            Dictionary with extracted summary and metadata
        """
        # Extract summary - everything up to keywords or the whole text
        summary = text_result
        if "keywords:" in text_result.lower():
            summary = text_result.split("keywords:", 1)[0].strip()
        
        # Try to extract keywords
        keywords = []
        keywords_match = re.search(r"keywords:(.+?)(?:speakers:|importance:|action_items:|key_quotes:|$)", 
                                  text_result.lower(), re.DOTALL)
        if keywords_match:
            keywords_text = keywords_match.group(1).strip()
            # Try to parse as list
            keywords = [k.strip() for k in re.split(r'[,\n•\-\[\]"]+', keywords_text) if k.strip()]
        
        # Extract speakers
        speakers = []
        speakers_match = re.search(r"speakers:(.+?)(?:importance:|action_items:|key_quotes:|$)", 
                                  text_result.lower(), re.DOTALL)
        if speakers_match:
            speakers_text = speakers_match.group(1).strip()
            speakers = [s.strip() for s in re.split(r'[,\n•\-\[\]"]+', speakers_text) if s.strip()]
        
        # Extract importance
        importance = 3  # Default mid-importance
        importance_match = re.search(r"importance:\s*(\d)", text_result.lower())
        if importance_match:
            try:
                importance = int(importance_match.group(1))
            except ValueError:
                pass
        
        # Extract action items
        action_items = []
        action_items_match = re.search(r"action_items:(.+?)(?:key_quotes:|$)", 
                                      text_result.lower(), re.DOTALL)
        if action_items_match:
            action_items_text = action_items_match.group(1).strip()
            action_items = [a.strip() for a in re.split(r'[•\-\[\]"]+', action_items_text) if a.strip()]
        
        # Extract key quotes
        key_quotes = []
        key_quotes_match = re.search(r"key_quotes:(.+?)$", text_result.lower(), re.DOTALL)
        if key_quotes_match:
            key_quotes_text = key_quotes_match.group(1).strip()
            key_quotes = [q.strip() for q in re.split(r'[•\-\[\]"]+', key_quotes_text) if q.strip()]
        
        return {
            "summary": summary,
            "keywords": keywords,
            "speakers": speakers,
            "importance": importance,
            "action_items": action_items,
            "key_quotes": key_quotes,
            "chunk_index": chunk_index,
            "position": position,
            "chunk_type": chunk_type
        }
    
    def summarize_chunk_sync(self, 
                           chunk: Dict[str, Any], 
                           document_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Synchronous version of summarize_chunk.
        
        Args:
            chunk: Chunk dictionary with text and metadata
            document_context: Optional context from document analysis
            
        Returns:
            Dictionary with summary and extracted metadata
        """
        import asyncio
        
        # Create an event loop and run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.summarize_chunk(chunk, document_context))
        finally:
            loop.close()