# This code shows how to properly handle async operations in Streamlit

import asyncio
import streamlit as st
from lean.options import ProcessingOptions
from lean.orchestrator import SummarizerFactory

def process_document_with_async_support():
    """Process document using async operations properly wrapped for Streamlit."""
    # Create progress indicators
    progress_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0)
    status_text = st.empty()
    
    try:
        # Get API key from environment
        import os
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            progress_placeholder.empty()
            status_text.empty()
            st.error("OpenAI API key not found! Please set the OPENAI_API_KEY environment variable.")
            return
        
        # Update progress
        progress_bar.progress(0.1)
        status_text.text("Creating processing pipeline...")
        
        # Initialize options with parallel processing but no caching
        options = ProcessingOptions(
            model_name=selected_model,
            temperature=temperature,
            min_chunks=min_chunks,
            detail_level=detail_level,
            enable_caching=False,  # Disable caching to avoid pickling issues
            max_concurrent_chunks=max_concurrent,
            passes=[],  # No passes for now - focus just on summarization
            user_instructions=user_instructions if user_instructions else None
        )
        
        # Create the pipeline
        pipeline = SummarizerFactory.create_pipeline(api_key=api_key, options=options)
        orchestrator = pipeline['orchestrator']
        
        # Define progress callback
        def update_progress(progress, message):
            progress_bar.progress(min(progress, 0.99))
            status_text.text(message)
        
        # Process document using asyncio event loop
        progress_bar.progress(0.2)
        status_text.text("Processing document...")
        
        # Run async process in a new event loop
        result = run_async_in_loop(
            orchestrator.process_document,  # async function
            document_text,                  # positional arg
            progress_callback=update_progress  # keyword arg
        )
        
        # Update progress
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        
        # Store results in session state
        st.session_state.summary_result = result
        st.session_state.processing_complete = True
        
        # Clean up progress indicators
        progress_placeholder.empty()
        status_text.empty()
        
        # Rerun to display results
        st.rerun()
        
    except Exception as e:
        # Clear progress indicators
        progress_placeholder.empty()
        status_text.empty()
        
        # Show error
        st.error(f"Error during processing: {str(e)}")
        st.exception(e)  # Show full exception

def run_async_in_loop(async_func, *args, **kwargs):
    """Run an async function in a new event loop and return the result."""
    # Create a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Run the async function and wait for completion
        result = loop.run_until_complete(async_func(*args, **kwargs))
        return result
    finally:
        # Always close the loop to avoid resource leaks
        loop.close()