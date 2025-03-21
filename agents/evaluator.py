# agents/evaluator.py
from typing import Dict, Any, List, Optional
from .base import BaseAgent

class EvaluatorAgent(BaseAgent):
    """
    Agent specialized in evaluating the importance, severity, or impact
    of identified items based on criteria in the configuration.
    """
    
    def __init__(
        self,
        llm_client,
        crew_type: str,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ):
        """
        Initialize an evaluator agent.
        
        Args:
            llm_client: LLM client for agent communication
            crew_type: Type of crew (issues, actions, opportunities)
            config: Optional pre-loaded configuration
            verbose: Whether to enable verbose mode
        """
        super().__init__(
            llm_client=llm_client,
            agent_type="evaluation",
            crew_type=crew_type,
            config=config,
            verbose=verbose
        )
    
    def evaluate_items(self, aggregated_items: Dict[str, Any], document_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate aggregated items for severity, importance, or impact.
        
        Args:
            aggregated_items: Items to evaluate
            document_info: Optional document metadata
            
        Returns:
            Evaluated items with ratings and rationales
        """
        # Get evaluation criteria from config
        criteria = self.config.get("evaluation", {}).get("criteria", {})
        
        # Prepare context for prompt building
        context = {
            "aggregated_items": aggregated_items,
            "document_info": document_info or {},
            "criteria": criteria
        }
        
        # Execute the evaluation task
        return self.execute_task(context=context)
    
    def get_ratings_scale(self) -> List[str]:
        """
        Get the appropriate ratings scale based on crew type.
        
        Returns:
            List of valid ratings from most to least severe/important
        """
        scales = {
            "issues": ["critical", "high", "medium", "low"],
            "actions": ["immediate", "high", "medium", "low"],
            "opportunities": ["strategic", "high", "medium", "minor"]
        }
        return scales.get(self.crew_type, ["high", "medium", "low"])