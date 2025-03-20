# ui_utils/refiner.py
"""
Summary refinement module for adjusting detail level of generated summaries.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class SummaryRefiner:
    """
    Handles refinement of generated summaries to adjust detail level or incorporate
    user feedback without reprocessing the entire document.
    """
    
    def __init__(self, llm_client):
        """
        Initialize the summary refiner.
        
        Args:
            llm_client: LLM client for summary refinement
        """
        self.llm_client = llm_client
    
    async def refine_summary(self, 
                           result: Dict[str, Any], 
                           refinement_type: str,
                           custom_instructions: Optional[str] = None) -> Dict[str, Any]:
        """
        Refine a summary based on the specified refinement type.
        
        Args:
            result: Original summary result dictionary
            refinement_type: Type of refinement to apply ("more_detail" or "more_concise")
            custom_instructions: Optional custom instructions for refinement
            
        Returns:
            Dictionary with refined summary and metadata
        """
        # Extract the current summary
        current_summary = result.get('summary', '')
        if not current_summary:
            logger.warning("Cannot refine: No summary found in result")
            return result
        
        # Get executive summary if available
        has_exec_summary = 'executive_summary' in result
        exec_summary = result.get('executive_summary', '')
        
        # Generate appropriate prompt based on refinement type
        if refinement_type == "more_detail":
            refined_summary = await self._add_more_detail(current_summary, custom_instructions)
            
            # Also refine executive summary if it exists
            refined_exec_summary = ""
            if has_exec_summary:
                refined_exec_summary = await self._add_more_detail(
                    exec_summary, 
                    custom_instructions,
                    is_executive=True
                )
        
        elif refinement_type == "more_concise":
            refined_summary = await self._make_more_concise(current_summary, custom_instructions)
            
            # Also refine executive summary if it exists
            refined_exec_summary = ""
            if has_exec_summary:
                refined_exec_summary = await self._make_more_concise(
                    exec_summary,
                    custom_instructions, 
                    is_executive=True
                )
        
        else:
            logger.warning(f"Unknown refinement type: {refinement_type}")
            return result
        
        # Create a copy of the result with the refined summary
        refined_result = result.copy()
        refined_result['summary'] = refined_summary
        
        # Update executive summary if it was refined
        if has_exec_summary:
            refined_result['executive_summary'] = refined_exec_summary
        
        # Add refinement metadata
        if 'metadata' not in refined_result:
            refined_result['metadata'] = {}
            
        refined_result['metadata']['refinement_applied'] = refinement_type
        if custom_instructions:
            refined_result['metadata']['custom_instructions'] = custom_instructions
        
        return refined_result
    
    async def _add_more_detail(self, 
                             summary: str, 
                             custom_instructions: Optional[str] = None,
                             is_executive: bool = False) -> str:
        """
        Add more detail to a summary.
        
        Args:
            summary: Current summary text
            custom_instructions: Optional custom instructions
            is_executive: Whether this is an executive summary
            
        Returns:
            More detailed summary
        """
        # Format summary type for prompt clarity
        summary_type = "executive summary" if is_executive else "summary"
        
        # Define default instructions for more detail
        default_instructions = """
            - Add more specific examples, data points, and context
            - Include more information about discussions, considerations, and rationales
            - Add more nuance and qualification to key points
            - Expand on implications and connections between topics
        """ if not is_executive else """
            - Add more context and background information
            - Include more key points while keeping it relatively concise
            - Provide more clarity on decisions and their strategic importance
        """
        
        # Use custom instructions if provided
        instructions = custom_instructions if custom_instructions else default_instructions
        
        # Create prompt for adding detail
        prompt = f"""
        Please expand the following {summary_type} with more details and context.
        
        INSTRUCTIONS:
        {instructions}
        
        IMPORTANT:
        - Maintain the same overall structure and key points
        - Do not contradict any information in the original {summary_type}
        - Do not fabricate specific facts, figures, or quotes not implied by the original
        - Keep the tone and style consistent with the original
        
        ORIGINAL {summary_type.upper()}:
        {summary}
        
        MORE DETAILED {summary_type.upper()}:
        """
        
        # Get more detailed summary from LLM
        try:
            detailed_summary = await self.llm_client.generate_completion_async(prompt)
            return detailed_summary
        except Exception as e:
            logger.error(f"Error adding detail to summary: {e}")
            return summary
    
    async def _make_more_concise(self, 
                               summary: str, 
                               custom_instructions: Optional[str] = None,
                               is_executive: bool = False) -> str:
        """
        Make a summary more concise.
        
        Args:
            summary: Current summary text
            custom_instructions: Optional custom instructions
            is_executive: Whether this is an executive summary
            
        Returns:
            More concise summary
        """
        # Format summary type for prompt clarity
        summary_type = "executive summary" if is_executive else "summary"
        
        # Define default instructions for more concise
        default_instructions = """
            - Remove redundant information and repetitive content
            - Focus on the most important points and insights
            - Reduce unnecessary details while preserving key information
            - Simplify complex explanations where possible
        """ if not is_executive else """
            - Focus only on the most critical insights and decisions
            - Remove all but the most essential context
            - Make every sentence deliver high-value information
        """
        
        # Use custom instructions if provided
        instructions = custom_instructions if custom_instructions else default_instructions
        
        # Create prompt for making more concise
        prompt = f"""
        Please make the following {summary_type} more concise while preserving key information.
        
        INSTRUCTIONS:
        {instructions}
        
        IMPORTANT:
        - Maintain all key points and insights
        - Do not omit critical information or decisions
        - Keep the tone and style consistent with the original
        - Focus on efficiency of expression, not just cutting content
        
        ORIGINAL {summary_type.upper()}:
        {summary}
        
        CONCISE {summary_type.upper()}:
        """
        
        # Get more concise summary from LLM
        try:
            concise_summary = await self.llm_client.generate_completion_async(prompt)
            return concise_summary
        except Exception as e:
            logger.error(f"Error making summary more concise: {e}")
            return summary

    async def incorporate_user_instructions(self, 
                                         result: Dict[str, Any], 
                                         user_instructions: str) -> Dict[str, Any]:
        """
        Incorporate specific user instructions to refine the summary.
        
        Args:
            result: Original summary result dictionary
            user_instructions: Specific instructions from the user
            
        Returns:
            Dictionary with refined summary and metadata
        """
        # Extract the current summary
        current_summary = result.get('summary', '')
        if not current_summary:
            logger.warning("Cannot refine: No summary found in result")
            return result
        
        # Get executive summary if available
        has_exec_summary = 'executive_summary' in result
        exec_summary = result.get('executive_summary', '')
        
        # Create prompt for incorporating user instructions
        prompt = f"""
        Please refine the following summary according to these user instructions:
        
        USER INSTRUCTIONS:
        {user_instructions}
        
        IMPORTANT:
        - Follow the user's instructions precisely
        - Maintain key information and insights
        - Do not contradict information in the original summary
        - Do not fabricate specific facts, figures, or quotes not implied by the original
        
        ORIGINAL SUMMARY:
        {current_summary}
        
        REFINED SUMMARY:
        """
        
        # Get refined summary from LLM
        try:
            refined_summary = await self.llm_client.generate_completion_async(prompt)
            
            # Also refine executive summary if it exists
            refined_exec_summary = ""
            if has_exec_summary:
                exec_prompt = f"""
                Please refine the following executive summary according to these user instructions:
                
                USER INSTRUCTIONS:
                {user_instructions}
                
                IMPORTANT:
                - Follow the user's instructions precisely
                - Maintain key information and insights
                - Keep it appropriately concise for an executive summary
                
                ORIGINAL EXECUTIVE SUMMARY:
                {exec_summary}
                
                REFINED EXECUTIVE SUMMARY:
                """
                
                refined_exec_summary = await self.llm_client.generate_completion_async(exec_prompt)
            
            # Create a copy of the result with the refined summary
            refined_result = result.copy()
            refined_result['summary'] = refined_summary
            
            # Update executive summary if it was refined
            if has_exec_summary:
                refined_result['executive_summary'] = refined_exec_summary
            
            # Add refinement metadata
            if 'metadata' not in refined_result:
                refined_result['metadata'] = {}
                
            refined_result['metadata']['user_instructions_applied'] = user_instructions
            
            return refined_result
            
        except Exception as e:
            logger.error(f"Error incorporating user instructions: {e}")
            return result