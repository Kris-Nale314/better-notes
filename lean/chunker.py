"""
Streamlined document chunking module with simplified, effective chunking strategies.
"""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class DocumentChunker:
    """
    Streamlined text chunking focused on practical effectiveness for transcripts and documents.
    This version simplifies the original DocumentChunker by focusing on the most effective
    chunking strategies while reducing complexity.
    """
    
    def __init__(self):
        """Initialize the document chunker."""
        pass
    
    def chunk_document(self, 
                      text: str, 
                      min_chunks: int = 3, 
                      max_chunk_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Divide a document into chunks based on content and structure.
        
        Args:
            text: Document text
            min_chunks: Minimum number of chunks to create
            max_chunk_size: Maximum size of each chunk in characters
                (if None, calculated based on min_chunks)
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Calculate max_chunk_size if not provided
        if max_chunk_size is None:
            # Total length divided by chunks, with 20% buffer
            max_chunk_size = int((len(text) / min_chunks) * 1.2)
            # Cap at reasonable maximum (approx. 4000 tokens)
            max_chunk_size = min(max_chunk_size, 16000)
        
        logger.info(f"Chunking document ({len(text)} chars) with target of {min_chunks} chunks, "
                   f"max_chunk_size={max_chunk_size}")
        
        # Handle single-chunk case
        if len(text) <= max_chunk_size and min_chunks <= 1:
            return [{
                'index': 0,
                'text': text,
                'start_pos': 0,
                'end_pos': len(text),
                'chunk_type': 'full_document',
                'position': 'full_document'
            }]
        
        # Detect if this is a transcript-like document
        is_transcript = self._is_transcript_like(text)
        
        # Choose chunking strategy based on document type
        if is_transcript:
            chunks = self._chunk_transcript(text, min_chunks, max_chunk_size)
            logger.info(f"Used transcript chunking strategy: {len(chunks)} chunks created")
        else:
            chunks = self._chunk_document_by_structure(text, min_chunks, max_chunk_size)
            logger.info(f"Used structure chunking strategy: {len(chunks)} chunks created")
        
        # Fallback if we didn't get enough chunks
        if len(chunks) < min_chunks:
            logger.info(f"Insufficient chunks ({len(chunks)}), using content-based chunking")
            chunks = self._chunk_by_content(text, min_chunks, max_chunk_size)
        
        # Ensure each chunk has an index
        for i, chunk in enumerate(chunks):
            chunk['index'] = i
        
        # Add position metadata
        self._add_position_metadata(chunks)
        
        logger.info(f"Final chunking: {len(chunks)} chunks")
        return chunks
    
    def _is_transcript_like(self, text: str) -> bool:
        """
        Detect if text appears to be a transcript or conversation.
        
        Args:
            text: Text to analyze
            
        Returns:
            Boolean indicating if the text is transcript-like
        """
        # Analyze a sample for efficiency
        sample = text[:min(len(text), 3000)]
        
        # Quick patterns that strongly indicate a transcript
        transcript_indicators = [
            # Speaker patterns
            r'^\s*[A-Z][a-z]+:',                # "John:"
            r'^\s*[A-Z][a-z]+ [A-Z][a-z]+:',    # "John Smith:"
            r'\n\s*[A-Z][a-z]+:',               # newline + "John:"
            r'\n\s*[A-Z][a-z]+ [A-Z][a-z]+:',   # newline + "John Smith:"
            
            # Time patterns
            r'\d{1,2}:\d{2}(:\d{2})?\s*[AP]M',  # 10:30 AM
            r'\[\d{1,2}:\d{2}(:\d{2})?\]',      # [10:30]
            
            # Meeting indicators
            r'meeting transcript',
            r'call transcript',
            r'attendees:',
            r'participants:'
        ]
        
        # Count matches for each pattern
        match_count = 0
        for pattern in transcript_indicators:
            matches = re.findall(pattern, sample, re.IGNORECASE | re.MULTILINE)
            match_count += len(matches)
            
            # Short-circuit if we have strong evidence
            if len(matches) >= 5:
                return True
        
        # If we have several matches across patterns, likely a transcript
        return match_count >= 8
    
    def _chunk_transcript(self, text: str, min_chunks: int, max_chunk_size: int) -> List[Dict[str, Any]]:
        """
        Chunk a transcript based on speaker turns and timestamps.
        
        Args:
            text: Document text
            min_chunks: Minimum number of chunks to create
            max_chunk_size: Maximum size of each chunk
            
        Returns:
            List of chunk dictionaries
        """
        # Find all potential speaker transitions
        speaker_patterns = [
            r'(^|\n)\s*([A-Z][a-z]+ ?[A-Z]?[a-z]*):',  # Speaker name followed by colon
            r'(^|\n)\s*([A-Z][a-z]+ [A-Z][a-z]+):',    # Full name followed by colon
            r'(^|\n)\s*\d{1,2}:\d{2}(:\d{2})?\s*[AP]M'  # Timestamp (e.g., 10:30 AM)
        ]
        
        # Find all matches for all patterns
        transitions = []
        for pattern in speaker_patterns:
            for match in re.finditer(pattern, text):
                transitions.append(match.start())
        
        # Sort transitions by position and remove duplicates
        transitions = sorted(set(transitions))
        
        # If too few transitions, fall back to content-based chunking
        if len(transitions) < min_chunks:
            return self._chunk_by_content(text, min_chunks, max_chunk_size)
        
        # Calculate target chunk size based on min_chunks
        target_size = len(text) / min_chunks
        
        # Create chunks based on transitions
        chunks = []
        chunk_start = 0
        current_size = 0
        
        for pos in transitions:
            # Skip very beginning positions
            if pos < 20:
                continue
                
            current_size = pos - chunk_start
            
            # Create a chunk if we're near target size or exceeding max
            if current_size >= target_size * 0.8 or current_size >= max_chunk_size:
                chunks.append({
                    'text': text[chunk_start:pos],
                    'start_pos': chunk_start,
                    'end_pos': pos,
                    'chunk_type': 'transcript_segment'
                })
                chunk_start = pos
        
        # Add the final chunk
        if chunk_start < len(text):
            chunks.append({
                'text': text[chunk_start:],
                'start_pos': chunk_start,
                'end_pos': len(text),
                'chunk_type': 'transcript_segment'
            })
        
        return chunks
    
    def _chunk_document_by_structure(self, text: str, min_chunks: int, max_chunk_size: int) -> List[Dict[str, Any]]:
        """
        Chunk document based on structural elements (sections, paragraphs).
        Simplified from the original implementation to focus on most effective patterns.
        
        Args:
            text: Document text
            min_chunks: Minimum number of chunks to create
            max_chunk_size: Maximum size of each chunk
            
        Returns:
            List of chunk dictionaries
        """
        # Find structural breaks with their priority
        # (pattern, strength) where strength is 0.0-1.0
        structure_patterns = [
            # Headers
            (r'(^|\n)#{1,3}\s+[^\n]+', 0.9),  # Markdown headers
            (r'(^|\n)[A-Z][A-Z\s]{3,}[A-Z](\s|:|\n)', 0.8),  # ALL CAPS HEADERS
            
            # Section breaks
            (r'\n\s*\n', 0.7),  # Paragraph breaks
            (r'\n\s*[-=*]{3,}\s*\n', 0.9),  # Horizontal rules
            
            # Lists
            (r'(^|\n)\s*\d+\.\s+', 0.5),  # Numbered lists
            (r'(^|\n)\s*[-*•]\s+', 0.5),  # Bullet points
        ]
        
        # Find potential break points
        break_points = []
        for pattern, strength in structure_patterns:
            for match in re.finditer(pattern, text):
                # Avoid breaks at the very beginning
                if match.start() > 20:
                    break_points.append((match.start(), strength))
        
        # If no break points found, fall back to content-based chunking
        if not break_points:
            return self._chunk_by_content(text, min_chunks, max_chunk_size)
        
        # Sort break points by position
        break_points.sort(key=lambda x: x[0])
        
        # Calculate desired chunk size
        target_size = len(text) / min_chunks
        
        # Create chunks
        chunks = []
        chunk_start = 0
        
        for pos, strength in break_points:
            chunk_size = pos - chunk_start
            
            # Create a chunk if size is appropriate
            if (chunk_size >= target_size * 0.7 and strength >= 0.7) or chunk_size >= max_chunk_size:
                chunks.append({
                    'text': text[chunk_start:pos],
                    'start_pos': chunk_start,
                    'end_pos': pos,
                    'chunk_type': 'structural_segment'
                })
                chunk_start = pos
        
        # Add the final chunk
        if chunk_start < len(text):
            chunks.append({
                'text': text[chunk_start:],
                'start_pos': chunk_start,
                'end_pos': len(text),
                'chunk_type': 'structural_segment'
            })
        
        return chunks
    
    def _chunk_by_content(self, text: str, min_chunks: int, max_chunk_size: int) -> List[Dict[str, Any]]:
        """
        Chunk document based on content, trying to break at sentence boundaries when possible.
        This is a simplified fallback method when structural chunking doesn't produce enough chunks.
        
        Args:
            text: Document text
            min_chunks: Minimum number of chunks to create
            max_chunk_size: Maximum size of each chunk
            
        Returns:
            List of chunk dictionaries
        """
        total_length = len(text)
        
        # Calculate target chunk size
        target_size = min(max_chunk_size, total_length / min_chunks)
        
        # Find all sentence boundaries
        sentence_breaks = []
        for match in re.finditer(r'[.!?]\s+', text):
            sentence_breaks.append(match.end())
        
        # If no sentence breaks found, just divide evenly
        if not sentence_breaks:
            chunks = []
            chunk_size = total_length // min_chunks
            
            for i in range(min_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, total_length)
                
                chunks.append({
                    'text': text[start:end],
                    'start_pos': start,
                    'end_pos': end,
                    'chunk_type': 'fixed_size_segment'
                })
            
            return chunks
        
        # Create chunks at sentence boundaries closest to target positions
        chunks = []
        current_pos = 0
        
        while current_pos < total_length:
            # Find target end position
            target_end = min(current_pos + target_size, total_length)
            
            # Find closest sentence boundary
            closest_boundary = None
            min_distance = float('inf')
            
            for boundary in sentence_breaks:
                if boundary <= current_pos:
                    continue
                
                distance = abs(boundary - target_end)
                if distance < min_distance:
                    closest_boundary = boundary
                    min_distance = distance
                
                # Stop searching if we're getting too far past target
                if boundary > target_end + (target_size * 0.3):
                    break
            
            # If no suitable boundary found or it's too far, just use target end
            if closest_boundary is None or min_distance > target_size * 0.3:
                end_pos = min(current_pos + int(target_size), total_length)
            else:
                end_pos = closest_boundary
            
            chunks.append({
                'text': text[current_pos:end_pos],
                'start_pos': current_pos,
                'end_pos': end_pos,
                'chunk_type': 'content_segment'
            })
            
            current_pos = end_pos
            
            # Break if we've created enough chunks
            if len(chunks) >= min_chunks and current_pos > total_length * 0.95:
                # Add the remainder as the final chunk
                if current_pos < total_length:
                    chunks.append({
                        'text': text[current_pos:],
                        'start_pos': current_pos,
                        'end_pos': total_length,
                        'chunk_type': 'content_segment'
                    })
                break
        
        return chunks
    
    def _add_position_metadata(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add position metadata to chunks (introduction, body, conclusion).
        
        Args:
            chunks: List of chunk dictionaries to modify in place
        """
        chunk_count = len(chunks)
        
        if chunk_count == 1:
            chunks[0]['position'] = 'full_document'
            return
        
        if chunk_count == 2:
            chunks[0]['position'] = 'introduction'
            chunks[1]['position'] = 'conclusion'
            return
        
        # For 3+ chunks, assign positions
        chunks[0]['position'] = 'introduction'
        chunks[-1]['position'] = 'conclusion'
        
        if chunk_count <= 4:
            # For 3-4 chunks, just have intro, middle, conclusion
            for i in range(1, chunk_count - 1):
                chunks[i]['position'] = 'middle'
        else:
            # For 5+ chunks, use early, middle, late positions
            early_threshold = max(1, chunk_count // 4)
            late_threshold = max(chunk_count - early_threshold - 1, chunk_count * 3 // 4)
            
            for i in range(1, chunk_count - 1):
                if i < early_threshold:
                    chunks[i]['position'] = 'early'
                elif i >= late_threshold:
                    chunks[i]['position'] = 'late'
                else:
                    chunks[i]['position'] = 'middle'