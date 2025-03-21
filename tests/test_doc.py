"""
Test script for the improved DocumentAnalyzer implementation.
"""

import sys
import os
from pathlib import Path
import time
import asyncio
import json

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lean.document import DocumentAnalyzer
from lean.async_openai_adapter import AsyncOpenAIAdapter

# Reuse the test document loading functions from test_chunker_v2.py
from test_chunker import load_test_document

async def test_document_analysis():
    """Test document analysis on a sample document."""
    print("\n--- Testing Document Analysis ---")
    
    # Load test document
    document = load_test_document("document_sample.txt")
    print(f"Loaded document: {len(document)} chars")
    
    # Create LLM client for testing
    # Note: This requires an OpenAI API key to be set in the environment
    llm_client = AsyncOpenAIAdapter(model="gpt-3.5-turbo")
    
    # Create analyzer
    analyzer = DocumentAnalyzer(llm_client)
    
    # Test basic stats
    start_time = time.time()
    stats = analyzer.get_basic_stats(document)
    elapsed = time.time() - start_time
    
    print(f"Basic stats computed in {elapsed:.3f} seconds:")
    print(f"  Word count: {stats['word_count']}")
    print(f"  Sentence count: {stats['sentence_count']}")
    print(f"  Paragraph count: {stats['paragraph_count']}")
    print(f"  Estimated tokens: {stats['estimated_tokens']}")
    
    # Test preview analysis
    print("\nAnalyzing document preview...")
    start_time = time.time()
    result = await analyzer.analyze_preview(document)
    elapsed = time.time() - start_time
    
    print(f"Document analysis completed in {elapsed:.3f} seconds")
    
    # Display analysis results
    summary = result.get('preview_analysis', {}).get('summary', 'No summary available')
    print(f"Summary: {summary}")
    
    topics = result.get('preview_analysis', {}).get('key_topics', [])
    print(f"Key topics: {', '.join(topics)}")
    
    domains = result.get('preview_analysis', {}).get('domain_categories', [])
    print(f"Domain categories: {', '.join(domains)}")
    
    is_transcript = result.get('is_meeting_transcript', False)
    print(f"Detected as transcript: {is_transcript}")
    
    return result

async def test_transcript_analysis():
    """Test document analysis on a sample transcript."""
    print("\n--- Testing Transcript Analysis ---")
    
    # Load test transcript
    transcript = load_test_document("transcript_sample.txt")
    print(f"Loaded transcript: {len(transcript)} chars")
    
    # Create LLM client for testing
    llm_client = AsyncOpenAIAdapter(model="gpt-3.5-turbo")
    
    # Create analyzer
    analyzer = DocumentAnalyzer(llm_client)
    
    # Test preview analysis
    print("\nAnalyzing transcript preview...")
    start_time = time.time()
    result = await analyzer.analyze_preview(transcript)
    elapsed = time.time() - start_time
    
    print(f"Transcript analysis completed in {elapsed:.3f} seconds")
    
    # Display analysis results
    summary = result.get('preview_analysis', {}).get('summary', 'No summary available')
    print(f"Summary: {summary}")
    
    purpose = result.get('preview_analysis', {}).get('meeting_purpose', 'No purpose specified')
    print(f"Meeting purpose: {purpose}")
    
    participants = result.get('preview_analysis', {}).get('participants', [])
    print(f"Participants: {', '.join(participants)}")
    
    is_transcript = result.get('is_meeting_transcript', False)
    print(f"Detected as transcript: {is_transcript}")
    
    return result

async def test_error_handling():
    """Test error handling in document analyzer."""
    print("\n--- Testing Error Handling ---")
    
    # Create a mock LLM client that raises an exception
    class MockLLMClient:
        async def generate_completion_async(self, prompt):
            raise Exception("Simulated API error")
    
    # Create analyzer with mock client
    analyzer = DocumentAnalyzer(MockLLMClient())
    
    # Test error handling
    print("Testing with simulated API error...")
    document = "This is a test document for error handling."
    
    try:
        result = await analyzer.analyze_preview(document)
        print("Error handled successfully")
        
        # Check fallback response
        summary = result.get('preview_analysis', {}).get('summary', '')
        print(f"Fallback summary: {summary}")
        
        basic_stats = result.get('basic_stats', {})
        print(f"Basic stats still available: {basic_stats.get('word_count')} words")
        
        return True
    except Exception as e:
        print(f"Error was not handled properly: {e}")
        return False

async def test_json_decoding_fallback():
    """Test fallback when JSON decoding fails."""
    print("\n--- Testing JSON Decoding Fallback ---")
    
    # Create a mock LLM client that returns non-JSON
    class MockLLMClient:
        async def generate_completion_async(self, prompt):
            return """
            Here's my analysis:
            
            Summary: This is a document about project evaluation.
            
            Client Name: Alpha Corp
            
            Key Topics:
            - Budget issues
            - Timeline delays
            - Technical challenges
            
            Domain Categories:
            - Project Management
            - Finance
            
            Participants: John Smith, Jane Doe
            """
    
    # Create analyzer with mock client
    analyzer = DocumentAnalyzer(MockLLMClient())
    
    # Test fallback mechanism
    print("Testing with non-JSON response...")
    document = "This is a test document for JSON fallback."
    
    result = await analyzer.analyze_preview(document)
    print("JSON fallback handled successfully")
    
    # Check extracted fields
    summary = result.get('preview_analysis', {}).get('summary', '')
    print(f"Extracted summary: {summary}")
    
    client = result.get('preview_analysis', {}).get('client_name', '')
    print(f"Extracted client: {client}")
    
    topics = result.get('preview_analysis', {}).get('key_topics', [])
    print(f"Extracted topics: {topics}")
    
    return result

async def run_all_tests():
    """Run all test cases."""
    print("===== Testing DocumentAnalyzer =====")
    
    # Check if API key is available
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in environment variables.")
        print("Some tests will be skipped or may fail.")
    
    # Run tests that don't require real API calls
    await test_error_handling()
    await test_json_decoding_fallback()
    
    # Run tests that require API calls if key is available
    if api_key:
        await test_document_analysis()
        await test_transcript_analysis()
    else:
        print("\nSkipping tests that require OpenAI API key.")
    
    print("\n===== All tests completed =====")

if __name__ == "__main__":
    asyncio.run(run_all_tests())