"""
Streamlined document analyzer for quick initial assessment and metadata extraction.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Common business domains for categorization (simplified)
BUSINESS_DOMAINS = [
    "AI", "Data", "Cloud", "Security", "DevOps", "Analytics", 
    "Digital Transformation", "Strategy", "Product", "Marketing", 
    "Sales", "HR", "Operations", "Finance", "Legal"
]

class DocumentAnalyzer:
    """
    Analyzes documents to extract key information and metadata.
    This streamlined version focuses on the most valuable metadata extraction
    while reducing complexity.
    """
    
    def __init__(self, llm_client):
        """
        Initialize the document analyzer.
        
        Args:
            llm_client: LLM client for text analysis
        """
        self.llm_client = llm_client
    
    def get_basic_stats(self, text: str) -> Dict[str, Any]:
        """
        Extract basic statistics from text.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary of basic stats
        """
        # Character count
        chars_with_spaces = len(text)
        chars_without_spaces = len(re.sub(r'\s', '', text))
        
        # Word count - simplified regex
        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)
        
        # Sentence count (approximate)
        sentence_count = len(re.findall(r'[.!?]+\s+[A-Z]|[.!?]+$', text))
        
        # Paragraph count
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        # Estimate token count (rough approximation)
        estimated_tokens = len(text) // 4
        
        return {
            "char_count": chars_with_spaces,
            "char_count_no_spaces": chars_without_spaces,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "estimated_tokens": estimated_tokens
        }
    
    async def analyze_preview(self, text: str, preview_length: int = 2000) -> Dict[str, Any]:
        """
        Analyze the beginning of a document to extract context and metadata.
        Streamlined version with simpler error handling.
        
        Args:
            text: Document text
            preview_length: Number of characters to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Extract preview (beginning of document)
        preview = text[:min(len(text), preview_length)]
        
        # Create analysis prompt
        prompt = f"""
        Analyze the beginning of this document/transcript to extract key information.
        
        Return your analysis in JSON format with these fields:
        - summary: A 1-2 sentence summary of what this document appears to be about
        - client_name: The name of the client or company being discussed (if mentioned)
        - meeting_purpose: The apparent purpose of this meeting/document (if it's a transcript or meeting notes)
        - key_topics: A list of 3-5 main topics that appear to be discussed
        - domain_categories: A list of 2-3 business domains this document relates to (from this list: {", ".join(BUSINESS_DOMAINS)})
        - participants: Any people mentioned as participants (if it's a meeting)
        
        If any field cannot be determined, use null or an empty list as appropriate.
        
        DOCUMENT PREVIEW:
        {preview}
        """
        
        try:
            # Get analysis from LLM
            response = await self.llm_client.generate_completion_async(prompt)
            
            # Parse JSON response
            try:
                analysis_result = json.loads(response)
                logger.info("Successfully parsed document preview analysis")
            except json.JSONDecodeError:
                # Simple fallback if JSON parsing fails
                analysis_result = self._extract_basic_analysis(response)
                logger.warning("JSON parsing failed, using basic text extraction")
            
            # Add is_transcript flag
            is_transcript = self._is_likely_transcript(preview, analysis_result)
            
            return {
                "preview_analysis": analysis_result,
                "is_meeting_transcript": is_transcript,
                "preview_length": len(preview),
                "basic_stats": self.get_basic_stats(text)
            }
            
        except Exception as e:
            logger.error(f"Error in document preview analysis: {e}")
            # Minimal fallback
            return {
                "preview_analysis": {
                    "summary": "Document analysis could not be completed",
                    "key_topics": [],
                    "domain_categories": []
                },
                "is_meeting_transcript": self._is_likely_transcript(preview),
                "preview_length": len(preview),
                "basic_stats": self.get_basic_stats(text)
            }
    
    def _extract_basic_analysis(self, text_response: str) -> Dict[str, Any]:
        """
        Extract basic preview metadata when JSON parsing fails.
        Simplified version with more reliable text extraction.
        
        Args:
            text_response: Text response from LLM
            
        Returns:
            Dictionary with extracted preview analysis
        """
        # Default result
        result = {
            "summary": "",
            "client_name": None,
            "meeting_purpose": None,
            "key_topics": [],
            "domain_categories": [],
            "participants": []
        }
        
        # Extract summary
        summary_match = re.search(r'summary:?\s*(.+?)(?:\n|$)', text_response, re.IGNORECASE)
        if summary_match:
            result["summary"] = summary_match.group(1).strip()
        
        # Extract client name
        client_match = re.search(r'client_?name:?\s*(.+?)(?:\n|$)', text_response, re.IGNORECASE)
        if client_match:
            client = client_match.group(1).strip()
            if client.lower() not in ["null", "none", "n/a"]:
                result["client_name"] = client
        
        # Extract meeting purpose
        purpose_match = re.search(r'meeting_?purpose:?\s*(.+?)(?:\n|$)', text_response, re.IGNORECASE)
        if purpose_match:
            purpose = purpose_match.group(1).strip()
            if purpose.lower() not in ["null", "none", "n/a"]:
                result["meeting_purpose"] = purpose
        
        # Extract key topics
        topics_section = re.search(r'key_?topics:?(.*?)(?:domain|participants|$)', text_response, re.IGNORECASE | re.DOTALL)
        if topics_section:
            topic_text = topics_section.group(1)
            # Handle both list formats and comma-separated
            topics = re.findall(r'[-•*]\s*([^,\n]+)|"([^"]+)"|\'([^\']+)\'|([^,\n]+)', topic_text)
            # Flatten and clean the topics
            result["key_topics"] = [next(t for t in topic if t) for topic in topics if any(t)]
        
        # Extract domain categories
        domains_section = re.search(r'domain_?categories:?(.*?)(?:participants|$)', text_response, re.IGNORECASE | re.DOTALL)
        if domains_section:
            domain_text = domains_section.group(1)
            domains = re.findall(r'[-•*]\s*([^,\n]+)|"([^"]+)"|\'([^\']+)\'|([^,\n]+)', domain_text)
            # Flatten, clean, and filter the domains
            domain_candidates = [next(d for d in domain if d).strip() for domain in domains if any(d)]
            result["domain_categories"] = [d for d in domain_candidates 
                                        if any(domain.lower() in d.lower() for domain in BUSINESS_DOMAINS)]
        
        # Extract participants
        participants_section = re.search(r'participants:?(.*?)(?:$)', text_response, re.IGNORECASE | re.DOTALL)
        if participants_section:
            participants_text = participants_section.group(1)
            participants = re.findall(r'[-•*]\s*([^,\n]+)|"([^"]+)"|\'([^\']+)\'|([^,\n]+)', participants_text)
            result["participants"] = [next(p for p in participant if p).strip() for participant in participants if any(p)]
        
        return result
    
    def _is_likely_transcript(self, text: str, analysis_result: Optional[Dict[str, Any]] = None) -> bool:
        """
        Determine if text is likely a meeting transcript.
        Simplified implementation focusing on the most reliable indicators.
        
        Args:
            text: Text to analyze
            analysis_result: Optional analysis result from LLM
            
        Returns:
            Boolean indicating if text is likely a transcript
        """
        # Check for transcript patterns
        transcript_indicators = [
            # Speaker indicators (most reliable)
            r'\n\s*[A-Z][a-z]+:', 
            r'\n\s*[A-Z][a-z]+ [A-Z][a-z]+:',
            
            # Time markers
            r'\d{1,2}:\d{2}(:\d{2})?\s*[AP]M',
            
            # Meeting markers
            r'meeting (started|began|commenced)',
            r'transcript',
            r'(attendees|participants):'
        ]
        
        # Count matches for indicators
        indicator_count = 0
        for pattern in transcript_indicators:
            matches = re.findall(pattern, text)
            indicator_count += len(matches)
            
            # Short-circuit if we have strong evidence
            if len(matches) >= 3:
                return True
        
        # If we have several matches, likely a transcript
        if indicator_count >= 5:
            return True
        
        # Use analysis result if available
        if analysis_result:
            # If there are participants and a meeting purpose, likely a transcript
            if (analysis_result.get('participants') and 
                analysis_result.get('meeting_purpose') and 
                len(analysis_result.get('participants', [])) > 1):
                return True
            
            # If the summary mentions meeting terms, likely a transcript
            summary = analysis_result.get('summary', '').lower()
            transcript_keywords = ['meeting', 'call', 'discussion', 'conversation', 'transcript']
            if any(keyword in summary for keyword in transcript_keywords):
                return True
        
        return False