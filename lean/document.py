"""
Improved document analyzer for Better Notes.
Provides robust analysis of document content with enhanced error handling.
"""

import re
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

# Common business domains for categorization
BUSINESS_DOMAINS = [
    "AI", "Data", "Cloud", "Security", "DevOps", "Analytics", 
    "Digital Transformation", "Strategy", "Product", "Marketing", 
    "Sales", "HR", "Operations", "Finance", "Legal"
]

class DocumentAnalyzer:
    """
    Analyzes documents to extract key information and metadata.
    Enhanced version with robust error handling and JSON parsing.
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
        try:
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
        except Exception as e:
            logger.error(f"Error getting basic stats: {e}")
            # Return minimal stats in case of error
            return {
                "char_count": len(text),
                "word_count": len(text.split()),
                "paragraph_count": 1,
                "estimated_tokens": len(text) // 4
            }
    
    async def analyze_preview(self, text: str, preview_length: int = 2000) -> Dict[str, Any]:
        """
        Analyze the beginning of a document to extract context and metadata.
        Enhanced with better error handling and JSON parsing.
        
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
        
        IMPORTANT: Your response MUST be valid JSON format only. No other text, explanations, or markdown.
        """
        
        try:
            # Get analysis from LLM
            logger.info("Sending preview analysis request to LLM")
            response = await self.llm_client.generate_completion_async(prompt)
            
            # Parse JSON response with multiple fallback strategies
            analysis_result = self._parse_json_safely(response)
            logger.info("Successfully parsed document preview analysis")
            
            # Add is_transcript flag
            is_transcript = self._is_likely_transcript(preview, analysis_result)
            
            # Create final result
            result = {
                "preview_analysis": analysis_result,
                "is_meeting_transcript": is_transcript,
                "preview_length": len(preview),
                "basic_stats": self.get_basic_stats(text)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in document preview analysis: {e}")
            # Create fallback result
            fallback_result = {
                "preview_analysis": {
                    "summary": "Document analysis could not be completed due to an error",
                    "client_name": None,
                    "meeting_purpose": None,
                    "key_topics": [],
                    "domain_categories": [],
                    "participants": []
                },
                "is_meeting_transcript": self._is_likely_transcript(preview),
                "preview_length": len(preview),
                "basic_stats": self.get_basic_stats(text)
            }
            
            return fallback_result
    
    def _parse_json_safely(self, text: str) -> Dict[str, Any]:
        """
        Robust JSON parsing with multiple fallback strategies.
        
        Args:
            text: Text to parse as JSON
            
        Returns:
            Parsed JSON as dictionary
        """
        # Skip logging the full text to avoid cluttering logs
        logger.info(f"Parsing JSON response of length {len(text)}")
        
        # Try direct JSON parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Direct JSON parsing failed, trying fallback methods")
        
        # Try to extract JSON from markdown code blocks
        try:
            json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
            match = re.search(json_pattern, text)
            if match:
                return json.loads(match.group(1))
        except Exception:
            logger.warning("Markdown code block extraction failed")
        
        # Try to extract any JSON-like structure
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except Exception:
            logger.warning("JSON structure extraction failed")
        
        # Final fallback: manual extraction of key fields
        logger.warning("All JSON parsing methods failed, falling back to manual extraction")
        return self._extract_basic_analysis(text)
    
    def _extract_basic_analysis(self, text_response: str) -> Dict[str, Any]:
        """
        Extract basic preview metadata when JSON parsing fails.
        Uses regex to find keys and values in free text.
        
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
        summary_match = re.search(r'(?:summary|about)[:]\s*(.+?)(?:\n|$)', 
                                 text_response, re.IGNORECASE)
        if summary_match:
            result["summary"] = summary_match.group(1).strip()
        else:
            # Try to find first paragraph that isn't a field name
            lines = text_response.split('\n')
            for line in lines:
                if len(line) > 30 and not any(key in line.lower() for key in ["summary:", "client:", "purpose:", "topics:", "domain:", "participants:"]):
                    result["summary"] = line.strip()
                    break
        
        # Extract client name
        client_match = re.search(r'(?:client|company|organization)[_\s]*name:?\s*(.+?)(?:\n|$)', 
                                text_response, re.IGNORECASE)
        if client_match:
            client = client_match.group(1).strip()
            if client.lower() not in ["null", "none", "n/a"]:
                result["client_name"] = client
        
        # Extract meeting purpose
        purpose_match = re.search(r'(?:meeting|document)?[_\s]*purpose:?\s*(.+?)(?:\n|$)', 
                                 text_response, re.IGNORECASE)
        if purpose_match:
            purpose = purpose_match.group(1).strip()
            if purpose.lower() not in ["null", "none", "n/a"]:
                result["meeting_purpose"] = purpose
        
        # Extract key topics
        topics = []
        topics_section = re.search(r'(?:key_topics|topics|key topics|main topics):?(.*?)(?:domain|categories|participants|$)', 
                                  text_response, re.IGNORECASE | re.DOTALL)
        if topics_section:
            topic_text = topics_section.group(1)
            # Find list items (whether numbered, bulleted, or quoted)
            topic_matches = re.findall(r'[-•*\d.]\s*"?([^,\n"]+)"?|"([^"]+)"|\'([^\']+)\'|([^,\n]+)', topic_text)
            # Flatten and clean
            for match in topic_matches:
                topic = next((t for t in match if t), "").strip()
                if topic and topic.lower() not in ["null", "none", "n/a"]:
                    topics.append(topic)
            
            # Only keep a reasonable number
            result["key_topics"] = topics[:5]
        
        # Extract domain categories
        domains = []
        domains_section = re.search(r'(?:domain_categories|domains|categories):?(.*?)(?:participants|$)', 
                                   text_response, re.IGNORECASE | re.DOTALL)
        if domains_section:
            domain_text = domains_section.group(1)
            domain_matches = re.findall(r'[-•*\d.]\s*"?([^,\n"]+)"?|"([^"]+)"|\'([^\']+)\'|([^,\n]+)', domain_text)
            # Flatten, clean, and filter
            for match in domain_matches:
                domain = next((d for d in match if d), "").strip()
                if domain and domain.lower() not in ["null", "none", "n/a"]:
                    # Check if it's in the allowed domains list (case-insensitive)
                    if any(allowed.lower() == domain.lower() or allowed.lower() in domain.lower() 
                          for allowed in BUSINESS_DOMAINS):
                        domains.append(domain)
            
            # Only keep a reasonable number
            result["domain_categories"] = domains[:3]
        
        # Extract participants
        participants = []
        participants_section = re.search(r'(?:participants|attendees|speakers):?(.*?)(?:$)', 
                                        text_response, re.IGNORECASE | re.DOTALL)
        if participants_section:
            participant_text = participants_section.group(1)
            participant_matches = re.findall(r'[-•*\d.]\s*"?([^,\n"]+)"?|"([^"]+)"|\'([^\']+)\'|([^,\n]+)', participant_text)
            # Flatten and clean
            for match in participant_matches:
                participant = next((p for p in match if p), "").strip()
                if participant and participant.lower() not in ["null", "none", "n/a"]:
                    participants.append(participant)
            
            result["participants"] = participants
        
        return result
    
    def _is_likely_transcript(self, text: str, analysis_result: Optional[Dict[str, Any]] = None) -> bool:
        """
        Determine if text is likely a meeting transcript.
        Enhanced with more reliable patterns.
        
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
            r'^[A-Z][a-z]+ [A-Z][a-z]+:',
            
            # Common transcript markers
            r'transcript of',
            r'meeting transcript',
            r'call transcript',
            
            # Time markers
            r'\d{1,2}:\d{2}(:\d{2})?\s*[AP]M',
            r'\[\d{1,2}:\d{2}(:\d{2})?\]',
            
            # Meeting markers
            r'meeting (started|began|commenced)',
            r'(attendees|participants):'
        ]
        
        # Count matches for indicators
        indicator_count = 0
        for pattern in transcript_indicators:
            matches = re.findall(pattern, text, re.IGNORECASE)
            indicator_count += len(matches)
            
            # Short-circuit if we have strong evidence
            if len(matches) >= 3:
                return True
        
        # If we have several matches, likely a transcript
        if indicator_count >= 5:
            return True
        
        # Check more subtle patterns related to speakers
        # Count lines that look like speaker turns
        speaker_lines = re.findall(r'^\s*[A-Z][a-zA-Z\s]+:', text, re.MULTILINE)
        if len(speaker_lines) >= 3:
            return True
        
        # Use analysis result if available
        if analysis_result:
            # If there are participants and a meeting purpose, likely a transcript
            has_participants = (analysis_result.get('participants') and 
                               len(analysis_result.get('participants', [])) > 1)
            has_meeting_purpose = bool(analysis_result.get('meeting_purpose'))
            
            if has_participants and has_meeting_purpose:
                return True
            
            # If the summary mentions meeting terms, likely a transcript
            summary = analysis_result.get('summary', '').lower()
            transcript_keywords = ['meeting', 'call', 'discussion', 'conversation', 
                                 'transcript', 'conference']
            if any(keyword in summary for keyword in transcript_keywords):
                return True
        
        return False