"""
Document analyzer for quick initial assessment and metadata extraction.
"""

import re
import json
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

# Common business domains and topics for categorization
BUSINESS_DOMAINS = [
    "AI", "Data", "Data Management", "AI Governance", "Platforms", "AI Adoption", 
    "People", "Skills", "Strategy", "Technology", "Innovation", "Cloud", 
    "Analytics", "Machine Learning", "Digital Transformation", "Security",
    "Infrastructure", "DevOps", "Agile", "Product Management", "Customer Experience",
    "Finance", "Marketing", "Sales", "HR", "Legal", "Operations", "Supply Chain",
    "Risk Management", "Compliance", "Enterprise Architecture"
]

class DocumentAnalyzer:
    """Analyzes documents to extract key information and metadata."""
    
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
        # Character count (with and without spaces)
        chars_with_spaces = len(text)
        chars_without_spaces = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
        
        # Word count
        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)
        
        # Sentence count (approximate)
        sentence_patterns = [r'[.!?]+\s+[A-Z]', r'[.!?]+$']
        sentence_count = 0
        for pattern in sentence_patterns:
            sentence_count += len(re.findall(pattern, text))
        
        # Paragraph count
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        # Estimate token count (rough approximation)
        estimated_tokens = len(text) // 4
        
        # Get average word length
        if word_count > 0:
            avg_word_length = sum(len(word) for word in words) / word_count
        else:
            avg_word_length = 0
            
        return {
            "char_count": chars_with_spaces,
            "char_count_no_spaces": chars_without_spaces,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "estimated_tokens": estimated_tokens,
            "avg_word_length": round(avg_word_length, 1)
        }
    
    async def analyze_preview(self, text: str, preview_length: int = 2000) -> Dict[str, Any]:
        """
        Analyze the beginning of a document to extract context and metadata.
        
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
        - key_topics: A list of 3-7 main topics that appear to be discussed
        - domain_categories: A list of 2-4 business domains this document relates to (from this list: {", ".join(BUSINESS_DOMAINS)})
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
                result = json.loads(response)
                return {
                    "preview_analysis": result,
                    "is_meeting_transcript": self._is_likely_transcript(preview, result),
                    "preview_length": len(preview)
                }
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                logger.warning("Failed to parse preview analysis JSON. Using fallback extraction.")
                return self._fallback_preview_extraction(response, preview)
        except Exception as e:
            logger.error(f"Error in document preview analysis: {e}")
            # Provide a basic fallback
            return {
                "preview_analysis": {
                    "summary": "Document analysis could not be completed",
                    "client_name": None,
                    "meeting_purpose": None,
                    "key_topics": [],
                    "domain_categories": [],
                    "participants": []
                },
                "is_meeting_transcript": self._is_likely_transcript(preview),
                "preview_length": len(preview)
            }
    
    def _fallback_preview_extraction(self, text_response: str, preview: str) -> Dict[str, Any]:
        """
        Extract preview metadata when JSON parsing fails.
        
        Args:
            text_response: Text response from LLM
            preview: Document preview text
            
        Returns:
            Dictionary with extracted preview analysis
        """
        # Extract summary
        summary = ""
        if "summary:" in text_response.lower():
            summary_section = text_response.lower().split("summary:")[1].split("\n")[0]
            summary = summary_section.strip()
        
        # Extract client name
        client_name = None
        if "client_name:" in text_response.lower():
            client_section = text_response.lower().split("client_name:")[1].split("\n")[0]
            client_name = client_section.strip()
            if client_name.lower() in ["null", "none", "n/a"]:
                client_name = None
        
        # Extract meeting purpose
        meeting_purpose = None
        if "meeting_purpose:" in text_response.lower():
            purpose_section = text_response.lower().split("meeting_purpose:")[1].split("\n")[0]
            meeting_purpose = purpose_section.strip()
            if meeting_purpose.lower() in ["null", "none", "n/a"]:
                meeting_purpose = None
        
        # Extract key topics
        key_topics = []
        if "key_topics:" in text_response.lower():
            topics_text = text_response.lower().split("key_topics:")[1].split("domain_categories:")[0]
            topic_matches = re.findall(r'[-•*]?\s*([^,\n]+)', topics_text)
            key_topics = [t.strip() for t in topic_matches if t.strip()]
        
        # Extract domain categories
        domain_categories = []
        if "domain_categories:" in text_response.lower():
            domains_text = text_response.lower().split("domain_categories:")[1].split("participants:")[0]
            domain_matches = re.findall(r'[-•*]?\s*([^,\n]+)', domains_text)
            domain_categories = [d.strip() for d in domain_matches if d.strip()]
            # Filter to valid domains
            domain_categories = [d for d in domain_categories if any(domain.lower() in d.lower() for domain in BUSINESS_DOMAINS)]
        
        # Extract participants
        participants = []
        if "participants:" in text_response.lower():
            participants_text = text_response.lower().split("participants:")[1]
            participant_matches = re.findall(r'[-•*]?\s*([^,\n]+)', participants_text)
            participants = [p.strip() for p in participant_matches if p.strip()]
        
        return {
            "preview_analysis": {
                "summary": summary,
                "client_name": client_name,
                "meeting_purpose": meeting_purpose,
                "key_topics": key_topics,
                "domain_categories": domain_categories,
                "participants": participants
            },
            "is_meeting_transcript": self._is_likely_transcript(preview),
            "preview_length": len(preview)
        }
    
    def _is_likely_transcript(self, text: str, analysis_result: Optional[Dict[str, Any]] = None) -> bool:
        """
        Determine if text is likely a meeting transcript.
        
        Args:
            text: Text to analyze
            analysis_result: Optional analysis result from LLM
            
        Returns:
            Boolean indicating if text is likely a transcript
        """
        # Check for transcript patterns
        transcript_patterns = [
            # Time stamps
            r'\d{1,2}:\d{2}(:\d{2})?\s*[AP]M',
            # Speaker indicators
            r'^[A-Z][a-z]+:',
            r'^[A-Z][a-z]+ [A-Z][a-z]+:',
            # Meeting markers
            r'meeting (started|began|commenced)',
            r'call (started|began|commenced)',
            # Participant lists
            r'(attendees|participants|present):'
        ]
        
        # Check for patterns
        for pattern in transcript_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return True
        
        # If we have analysis, use it to help determine
        if analysis_result:
            # If there are participants and a meeting purpose, likely a transcript
            if (analysis_result.get('participants') and 
                analysis_result.get('meeting_purpose') and 
                len(analysis_result.get('participants', [])) > 1):
                return True
            
            # If the summary mentions "meeting", "call", "discussion", likely a transcript
            summary = analysis_result.get('summary', '').lower()
            transcript_keywords = ['meeting', 'call', 'discussion', 'conversation', 'transcript']
            if any(keyword in summary for keyword in transcript_keywords):
                return True
        
        return False