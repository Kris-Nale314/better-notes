# agents/formatter.py
from typing import Dict, Any, List, Optional
from .base import BaseAgent

class FormatterAgent(BaseAgent):
    """
    Agent specialized in formatting analysis results into a clear, structured report.
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
        Initialize a formatter agent.
        
        Args:
            llm_client: LLM client for agent communication
            crew_type: Type of crew (issues, actions, opportunities)
            config: Optional pre-loaded configuration
            verbose: Whether to enable verbose mode
            max_chunk_size: Maximum size of text chunks to process
        """
        super().__init__(
            llm_client=llm_client,
            agent_type="formatting",
            crew_type=crew_type,
            config=config,
            verbose=verbose,
            max_chunk_size=max_chunk_size, 
            max_rpm = max_rpm   # Pass this parameter to BaseAgent
        )
    
    # Rest of the implementation...
    
    def format_report(self, evaluated_items: Dict[str, Any], document_info: Optional[Dict[str, Any]] = None, 
                      user_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Format evaluated items into a structured report.
        
        Args:
            evaluated_items: Items to include in the report
            document_info: Optional document metadata
            user_preferences: Optional user formatting preferences
            
        Returns:
            Formatted report
        """
        # Get format template from config
        format_template = self.config.get("formatting", {}).get("format_template", "")
        
        # Prepare context for prompt building
        context = {
            "evaluated_items": evaluated_items,
            "document_info": document_info or {},
            "user_preferences": user_preferences or {},
            "format_template": format_template
        }
        
        # Execute the formatting task
        return self.execute_task(context=context)
    
    def get_section_title(self) -> str:
        """
        Get an appropriate section title based on the crew type.
        
        Returns:
            Section title string
        """
        titles = {
            "issues": "Issues Identified",
            "actions": "Action Items",
            "opportunities": "Opportunities Discovered"
        }
        return titles.get(self.crew_type, f"{self.crew_type.title()} Analysis")
    
    def get_default_sections(self) -> List[str]:
        """
        Get the default report sections based on crew type.
        
        Returns:
            List of section names
        """
        common_sections = ["Executive Summary", "Methodology"]
        
        crew_specific_sections = {
            "issues": ["Critical Issues", "High-Priority Issues", "Medium-Priority Issues", "Low-Priority Issues"],
            "actions": ["Immediate Actions", "Short-Term Actions", "Long-Term Actions"],
            "opportunities": ["Strategic Opportunities", "Tactical Improvements", "Future Considerations"]
        }
        
        return common_sections + crew_specific_sections.get(self.crew_type, ["Findings", "Analysis"])
    
    def apply_user_formatting(self, formatted_report: str, user_preferences: Dict[str, Any]) -> str:
        """
        Apply user-specific formatting preferences to the report.
        
        Args:
            formatted_report: Basic formatted report
            user_preferences: User formatting preferences
            
        Returns:
            Report with user formatting applied
        """
        # This is a placeholder for applying user preferences like:
        # - Including/excluding executive summary
        # - Adjusting detail level
        # - Emphasizing specific areas
        # - Custom organization
        
        # Would be implemented based on specific UI options
        return formatted_report