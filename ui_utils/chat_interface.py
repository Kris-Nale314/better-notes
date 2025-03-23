"""
Chat interface module for Better Notes.
Provides conversational interface components for document analysis.
"""

import streamlit as st
import asyncio
from typing import Dict, Any, List, Optional, Callable, Union

def initialize_chat_state():
    """Initialize session state variables for chat interface."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""

def display_chat_interface(
    llm_client, 
    document_text: str,
    summary_text: str,
    document_info: Optional[Dict[str, Any]] = None,
    on_new_message: Optional[Callable] = None
):
    """
    Display a chat interface for interacting with the document.
    
    Args:
        llm_client: LLM client for generating responses
        document_text: Original document text
        summary_text: Summary text generated from the document
        document_info: Optional document metadata
        on_new_message: Optional callback when new message is added
    """
    initialize_chat_state()
    
    st.divider()
    st.subheader("💬 Ask About This Document")
    
    # Chat message display container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        display_chat_messages(st.session_state.chat_history)
    
    # Quick question buttons
    display_quick_question_buttons(llm_client, document_text, summary_text, document_info, on_new_message)
    
    # User question input
    with st.form(key="chat_form", clear_on_submit=True):
        user_question = st.text_input(
            "Your question:",
            key="chat_input",
            placeholder="Ask a question about this document..."
        )
        submit_button = st.form_submit_button("Send")
        
        if submit_button and user_question:
            process_chat_question(llm_client, user_question, document_text, summary_text, document_info, on_new_message)

def display_chat_messages(chat_history: List[Dict[str, str]]):
    """
    Display chat messages from history.
    
    Args:
        chat_history: List of message dictionaries with 'role' and 'content'
    """
    for message in chat_history:
        role = message["role"]
        content = message["content"]
        
        # Choose styling based on role
        if role == "user":
            message_class = "chat-message user-message"
            prefix = "You: "
        else:
            message_class = "chat-message assistant-message"
            prefix = "Assistant: "
        
        st.markdown(f"""
            <div class="{message_class}">
                <strong>{prefix}</strong>{content}
            </div>
        """, unsafe_allow_html=True)

def display_quick_question_buttons(
    llm_client, 
    document_text: str,
    summary_text: str,
    document_info: Optional[Dict[str, Any]] = None,
    on_new_message: Optional[Callable] = None
):
    """
    Display quick question buttons for common questions.
    
    Args:
        llm_client: LLM client for generating responses
        document_text: Original document text
        summary_text: Summary text generated from the document
        document_info: Optional document metadata
        on_new_message: Optional callback when new message is added
    """
    st.markdown("<div style='display: flex; flex-wrap: wrap;'>", unsafe_allow_html=True)
    
    # Determine document type for contextual questions
    doc_type = "transcript" if document_info and document_info.get("is_meeting_transcript") else "document"
    
    # Select questions based on document type
    if doc_type == "transcript":
        quick_questions = [
            "What were the key decisions made?",
            "Summarize the main discussion points",
            "Who were the participants?",
            "What action items were mentioned?"
        ]
    else:
        quick_questions = [
            "Summarize this document",
            "What are the key points?",
            "Explain this in simpler terms",
            "What are the main recommendations?"
        ]
    
    # Create columns for the buttons
    cols = st.columns(4)
    for i, question in enumerate(quick_questions):
        with cols[i]:
            if st.button(question, key=f"quick_{i}"):
                process_chat_question(llm_client, question, document_text, summary_text, document_info, on_new_message)
    
    st.markdown("</div>", unsafe_allow_html=True)

def process_chat_question(
    llm_client, 
    question: str, 
    document_text: str,
    summary_text: str,
    document_info: Optional[Dict[str, Any]] = None,
    on_new_message: Optional[Callable] = None
):
    """
    Process a chat question and add to history.
    
    Args:
        llm_client: LLM client for generating responses
        question: User's question
        document_text: Original document text
        summary_text: Summary text generated from the document
        document_info: Optional document metadata
        on_new_message: Optional callback when new message is added
    """
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": question})
    
    # Call the callback if provided
    if on_new_message:
        on_new_message("user", question)
    
    # Create a placeholder for the assistant's response
    with st.spinner("Thinking..."):
        # Generate context
        truncated_doc = document_text[:3000]
        
        # Extract document type and metadata
        doc_type = "transcript" if document_info and document_info.get("is_meeting_transcript") else "document"
        
        # Create prompt
        prompt = f"""
        You are an AI assistant helping with document analysis and questions.
        
        DOCUMENT SUMMARY:
        {summary_text}
        
        DOCUMENT TYPE: {doc_type}
        
        DOCUMENT EXCERPT (beginning of document):
        {truncated_doc}
        
        USER QUESTION: {question}
        
        Please answer the question based on the document information provided.
        Focus on being helpful, concise, and accurate.
        If the information is not available in the context, say so.
        """
        
        # Get response from LLM
        try:
            # Use async calls if available, otherwise fall back to sync
            if hasattr(llm_client, 'generate_completion_async'):
                response = asyncio.run(llm_client.generate_completion_async(prompt))
            else:
                response = llm_client.generate_completion(prompt)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Call the callback if provided
            if on_new_message:
                on_new_message("assistant", response)
            
            # Force a rerun to update the display
            st.rerun()
        except Exception as e:
            # Handle error gracefully
            error_message = f"Sorry, I encountered an error while processing your question: {str(e)}"
            st.session_state.chat_history.append({"role": "assistant", "content": error_message})
            
            # Call the callback if provided
            if on_new_message:
                on_new_message("assistant", error_message)
                
            st.rerun()

def create_chat_export_button():
    """Create a button to export chat history."""
    if st.session_state.chat_history:
        chat_export = ""
        for message in st.session_state.chat_history:
            prefix = "You: " if message["role"] == "user" else "Assistant: "
            chat_export += f"{prefix}{message['content']}\n\n"
        
        st.download_button(
            "Export Conversation",
            data=chat_export,
            file_name="document_chat_export.txt",
            mime="text/plain",
            help="Download the conversation as a text file"
        )

def clear_chat_history():
    """Clear the chat history from session state."""
    if st.button("Clear Conversation", help="Clear the current conversation history"):
        st.session_state.chat_history = []
        st.rerun()