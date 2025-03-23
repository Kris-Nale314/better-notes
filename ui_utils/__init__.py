#
"""
UI utilities for Better Notes.
Provides styling, visualization, and interface components.
"""

from .core_styling import (
    apply_core_styles,
    apply_component_styles,
    apply_analysis_styles,
    highlight_theme
)

from .agent_visualization import (
    display_agent_cards,
    display_compact_agent_status,
    display_agent_progress,
    update_agent_status,
    create_agent_status_tracker
)

from .result_formatting import (
    enhance_result_display,
    format_log_entries,
    create_download_button
)

from .chat_interface import (
    initialize_chat_state,
    display_chat_interface,
    display_chat_messages,
    process_chat_question,
    create_chat_export_button,
    clear_chat_history
)

from .progress_tracking import (
    ProgressTracker,
    display_progress_bar,
    create_progress_callback,
    log_progress
)

# For backward compatibility with existing code
def apply_custom_css():
    """Legacy function for backward compatibility."""
    apply_component_styles()