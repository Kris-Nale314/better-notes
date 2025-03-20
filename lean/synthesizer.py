# lean/synthesizer.py
"""
Summary synthesizer that combines chunk summaries into a coherent whole.
"""

import logging
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class Synthesizer:
    """Synthesizes multiple chunk summaries into a coherent whole."""
    
    def __init__(self, llm_client):
        """
        Initialize the synthesizer.
        
        Args:
            llm_client: LLM client for generating summaries
        """
        self.llm_client = llm_client
    
    async def synthesize_summaries(self, 
                                 chunk_summaries: List[Dict[str, Any]],
                                 document_context: Optional[Dict[str, Any]] = None,
                                 detail_level: str = "detailed") -> Dict[str, Any]:
        """
        Synthesize chunk summaries into a coherent whole.
        
        Args:
            chunk_summaries: List of chunk summary results
            document_context: Document context information
            detail_level: Level of detail ('essential', 'detailed', 'detailed-complex')
            
        Returns:
            Dictionary with synthesized summary and metadata
        """
        # Ensure document_context is a dictionary
        if document_context is None:
            document_context = {}
        elif not isinstance(document_context, dict):
            logger.warning(f"document_context is not a dictionary: {type(document_context)}")
            document_context = {"original_text": document_context}
        
        # Log synthesis parameters
        logger.info(f"Synthesizing with detail level: {detail_level}")
        
        # Sort chunk results by index if not already sorted
        sorted_summaries = sorted(chunk_summaries, key=lambda x: x.get('chunk_index', 0))
        
        # Generate summary based on detail level
        if detail_level == "essential":
            return await self._generate_essential_notes(sorted_summaries, document_context)
        elif detail_level == "detailed-complex":
            return await self._generate_detailed_notes(sorted_summaries, document_context, is_complex=True)
        else:  # default to "detailed"
            return await self._generate_detailed_notes(sorted_summaries, document_context, is_complex=False)
    
    async def _generate_essential_notes(self, 
                                      chunk_summaries: List[Dict[str, Any]],
                                      document_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate concise, essential notes from chunk summaries.
        
        Args:
            chunk_summaries: Sorted list of chunk summaries
            document_context: Document context information
            
        Returns:
            Dictionary with synthesized summary and metadata
        """
        # Extract parameters from document context
        is_transcript = document_context.get('is_meeting_transcript', False)
        meeting_purpose = document_context.get('meeting_purpose', '')
        user_instructions = document_context.get('user_instructions', '')
        
        # Prepare combined text from chunk summaries
        combined_text = ""
        for i, chunk in enumerate(chunk_summaries):
            summary = chunk.get('summary', '')
            combined_text += f"SECTION {i+1}:\n{summary}\n\n"
        
        # Build prompt for essential summary
        prompt = f"""
        Create a concise, organized summary of the following document sections.
        
        IMPORTANT INSTRUCTIONS:
        1. Focus only on the most essential information and key points
        2. Use bullet points for clarity and brevity
        3. Group related information under logical headings
        4. Maintain the original terminology and technical terms
        
        {'DOCUMENT TYPE: Meeting Transcript' if is_transcript else 'DOCUMENT TYPE: Document'}
        {f'MEETING PURPOSE: {meeting_purpose}' if meeting_purpose else ''}
        {f'USER INSTRUCTIONS: {user_instructions}' if user_instructions else ''}
        
        DOCUMENT SECTIONS:
        {combined_text}
        
        First, write a brief executive summary (2-3 sentences).
        Then create an organized, bulleted summary focusing on the essential information.
        """
        
        # Generate summary using LLM
        try:
            # Generate full summary
            summary = await self.llm_client.generate_completion_async(prompt)
            
            # Generate a short executive summary
            exec_prompt = f"""
            Based on this summary, create a very concise executive summary (2-3 sentences maximum):
            
            {summary}
            """
            executive_summary = await self.llm_client.generate_completion_async(exec_prompt)
            
            # Return both summaries
            return {
                'summary': summary,
                'executive_summary': executive_summary
            }
            
        except Exception as e:
            logger.error(f"Error generating essential notes: {e}")
            # Return a basic result in case of error
            return {
                'summary': "Error generating summary. Please try again.",
                'error': str(e)
            }
    
    async def _generate_detailed_notes(self, 
                                     chunk_summaries: List[Dict[str, Any]],
                                     document_context: Dict[str, Any],
                                     is_complex: bool = False) -> Dict[str, Any]:
        """
        Generate detailed notes from chunk summaries.
        
        Args:
            chunk_summaries: Sorted list of chunk summaries
            document_context: Document context information
            is_complex: Whether to generate more complex notes
            
        Returns:
            Dictionary with synthesized summary and metadata
        """
        # Extract parameters from document context
        is_transcript = document_context.get('is_meeting_transcript', False)
        meeting_purpose = document_context.get('meeting_purpose', '')
        user_instructions = document_context.get('user_instructions', '')
        client_name = document_context.get('client_name', '')
        
        # Get any preview analysis if available
        preview_analysis = document_context.get('preview_analysis', {})
        key_topics = preview_analysis.get('key_topics', []) if isinstance(preview_analysis, dict) else []
        domain_categories = preview_analysis.get('domain_categories', []) if isinstance(preview_analysis, dict) else []
        
        # Prepare combined text from chunk summaries
        combined_text = ""
        for i, chunk in enumerate(chunk_summaries):
            summary = chunk.get('summary', '')
            position = chunk.get('position', '')
            combined_text += f"SECTION {i+1}"
            if position:
                combined_text += f" ({position.upper()})"
            combined_text += f":\n{summary}\n\n"
        
        # Adjust detail instruction based on complexity
        if is_complex:
            detail_instruction = """
            Create a comprehensive, detailed summary with:
            - Thorough coverage of all significant information
            - Multiple logical sections organized by topic
            - Rich context and specific details preserved
            - Special attention to relationships between ideas
            """
        else:
            detail_instruction = """
            Create a well-balanced summary with:
            - All important information included
            - Logical organization by topic or theme
            - Key details and context where relevant
            - Clear structure with appropriate headings
            """
        
        # Build prompt for detailed summary
        prompt = f"""
        {detail_instruction}
        
        IMPORTANT INSTRUCTIONS:
        1. Organize information into logical categories based on content
        2. Use bullet points for clarity and readability
        3. Include specific details, numbers, and examples where important
        4. Maintain original terminology and phrasing when possible
        5. Include an executive summary at the beginning (3-5 sentences)
        
        {'DOCUMENT TYPE: Meeting Transcript' if is_transcript else 'DOCUMENT TYPE: Document'}
        {f'MEETING PURPOSE: {meeting_purpose}' if meeting_purpose else ''}
        {f'CLIENT: {client_name}' if client_name else ''}
        {f'USER INSTRUCTIONS: {user_instructions}' if user_instructions else ''}
        
        {f'KEY TOPICS: {", ".join(key_topics)}' if key_topics else ''}
        {f'DOMAINS: {", ".join(domain_categories)}' if domain_categories else ''}
        
        DOCUMENT SECTIONS:
        {combined_text}
        
        First, write an executive summary (3-5 sentences).
        Then create a well-structured, detailed summary with appropriate headings and bullet points.
        """
        
        # Generate summary using LLM
        try:
            # Generate the main summary
            summary = await self.llm_client.generate_completion_async(prompt)
            
            # Extract the executive summary
            executive_summary = ""
            if "# Executive Summary" in summary:
                # Try to extract the executive summary section
                parts = summary.split("# Executive Summary", 1)
                if len(parts) > 1:
                    exec_part = parts[1].split("#", 1)[0].strip()
                    executive_summary = exec_part
            
            # If no executive summary found, generate one
            if not executive_summary:
                exec_prompt = f"""
                Based on this detailed summary, create a concise executive summary (3-5 sentences):
                
                {summary}
                """
                executive_summary = await self.llm_client.generate_completion_async(exec_prompt)
            
            # Return results
            return {
                'summary': summary,
                'executive_summary': executive_summary
            }
            
        except Exception as e:
            logger.error(f"Error generating detailed notes: {e}")
            # Return a basic result in case of error
            return {
                'summary': "Error generating detailed summary. Please try again.",
                'error': str(e)
            }