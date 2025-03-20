# lean/options.py
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ProcessingOptions:
    """Configuration options for document processing."""

    # Model configuration
    model_name: str = "gpt-3.5-turbo"  # Default model
    temperature: float = 0.2  # Default temperature

    # Chunking parameters
    min_chunks: int = 3
    max_chunk_size: Optional[int] = None

    # Analysis options
    preview_length: int = 2000

    # Summary options
    detail_level: str = "detailed"  # 'essential', 'detailed', 'detailed-complex'
    include_action_items: bool = True  # No longer used directly, but kept for compatibility

    # Performance options
    max_concurrent_chunks: int = 5
    enable_caching: bool = True       # NEW: Enable/disable caching
    cache_dir: str = ".cache"       # NEW: Cache directory

    # Output options
    include_metadata: bool = True

    # User guidance
    user_instructions: Optional[str] = None

    # Passes to run
    passes: List[str] = None  # List of pass types to run