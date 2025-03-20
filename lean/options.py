# lean/options.py
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Tuple

@dataclass
class ProcessingOptions:
    """
    Configuration options for document processing and summarization.

    This class uses dataclasses for concise definition and easy initialization.
    It includes options for model selection, chunking, summarization,
    performance, and pass selection.
    """

    # --- Core Model Settings ---
    model_name: str = "gpt-3.5-turbo"  # Default model
    temperature: float = 0.2  # Default temperature for consistent results

    # --- Chunking Settings ---
    min_chunks: int = 3  # Minimum number of chunks
    max_chunk_size: Optional[int] = None  # Auto-calculated if None

    # --- Summarization Settings ---
    detail_level: str = "detailed"  # Options: "essential", "detailed", "detailed-complex"

    # --- Analysis/Preview Settings ---
    preview_length: int = 2000 # Number of characters for document analysis

    # --- Performance Settings ---
    max_concurrent_chunks: int = 5  # Max concurrent chunk processing
    enable_caching: bool = True  # Enable/disable caching
    cache_dir: str = ".cache"  # Cache directory

    # --- Output Settings ---
    include_metadata: bool = True  # Include metadata in the output

    # --- User Instructions ---
    user_instructions: Optional[str] = None  # Custom instructions for the LLM

    # --- Pass Selection ---
    passes: List[str] = field(default_factory=list)  # List of pass types to run

    # --- Pass-Specific Options (NEW) ---
    pass_options: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    #  Example:
    #  pass_options = {
    #     "issue_identification": {"severity_threshold": "medium"},
    #     "opportunity_identification": {"min_impact": "high"}
    #  }

    # --- Refinement options --- #Future use
    refinement_type: Optional[str] = None
    custom_instructions: Optional[str] = None

    def get_pass_options(self, pass_type: str) -> Dict[str, Any]:
        """
        Retrieves the pass-specific options for a given pass type.

        Args:
            pass_type: The type of the pass (e.g., "issue_identification").

        Returns:
            A dictionary of options for the specified pass, or an empty
            dictionary if no options are defined for that pass.
        """
        return self.pass_options.get(pass_type, {})