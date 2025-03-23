"""
Test script for the InstructorAgent.
This script verifies that the InstructorAgent can properly create tailored instructions
for different agents based on document context and user preferences.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add parent directory to path so we can import project modules
sys.path.append(str(Path(__file__).parent.parent))

# Import the InstructorAgent and required components
from agents.instructor import InstructorAgent
from lean.async_openai_adapter import AsyncOpenAIAdapter

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_test_config():
    """Load configuration for testing."""
    config_path = Path(__file__).parent.parent / "agents" / "config" / "issues_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        logger.warning(f"Config file not found at {config_path}. Using empty config.")
        return {}

def create_test_document_info():
    """Create a test document info object."""
    return {
        "is_meeting_transcript": False,
        "original_text_length": 5000,
        "preview_analysis": {
            "summary": "This document describes a software project with various technical challenges.",
            "key_topics": ["software development", "technical challenges", "project planning", 
                          "resource allocation", "timeline"],
            "domain_categories": ["Technology", "Project Management"]
        },
        "basic_stats": {
            "word_count": 1200,
            "paragraph_count": 25,
            "sentence_count": 75,
            "char_count": 7500,
            "estimated_tokens": 1800
        }
    }

def create_test_user_preferences():
    """Create test user preferences."""
    return {
        "detail_level": "standard",
        "focus_areas": ["Technical", "Risk"],
        "user_instructions": "Focus on identifying technical issues and security vulnerabilities."
    }

def print_instructions(instructions):
    """Pretty print the instructions."""
    print("\n==== GENERATED INSTRUCTIONS ====\n")
    for agent_type, agent_instructions in instructions.items():
        print(f"\n--- {agent_type.upper()} AGENT ---")
        
        if isinstance(agent_instructions, dict):
            if "instructions" in agent_instructions:
                print(f"Instructions: {agent_instructions['instructions']}")
            if "emphasis" in agent_instructions:
                print(f"Emphasis: {agent_instructions['emphasis']}")
        else:
            print(f"Instructions: {agent_instructions}")
        
        print("-" * 50)

def main():
    """Run the test."""
    # Check for OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return
    
    logger.info("Starting InstructorAgent test")
    
    # Load test config
    config = load_test_config()
    
    # Create LLM client
    llm_client = AsyncOpenAIAdapter(
        model="gpt-3.5-turbo",  # Use 3.5 to save costs during testing
        api_key=api_key,
        temperature=0.2
    )
    
    # Create InstructorAgent
    instructor = InstructorAgent(
        llm_client=llm_client,
        config=config,
        verbose=True,
        max_chunk_size=1500,
        max_rpm=10
    )
    
    # Create test data
    document_info = create_test_document_info()
    user_preferences = create_test_user_preferences()
    
    # Test with different crew types
    crew_types = ["issues"]
    
    for crew_type in crew_types:
        logger.info(f"Testing InstructorAgent with crew_type: {crew_type}")
        
        # Create instructions
        try:
            instructions = instructor.create_instructions(
                document_info=document_info,
                user_preferences=user_preferences,
                crew_type=crew_type
            )
            
            # Print the results
            print_instructions(instructions)
            
            # Save results to file
            output_dir = Path(__file__).parent / "outputs"
            output_dir.mkdir(exist_ok=True)
            
            with open(output_dir / f"{crew_type}_instructions.json", 'w') as f:
                json.dump(instructions, f, indent=2)
            
            logger.info(f"Test for {crew_type} completed successfully!")
            
        except Exception as e:
            logger.error(f"Test for {crew_type} failed: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()