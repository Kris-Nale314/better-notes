"""
Test script for the improved DocumentChunkerV2 implementation.
"""

import sys
import os
from pathlib import Path
import time

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lean.chunker import DocumentChunker

def load_test_document(filename):
    """Load a test document from the test_docs directory."""
    test_dir = Path(__file__).parent / "test_docs"
    
    # Create the directory if it doesn't exist
    test_dir.mkdir(exist_ok=True)
    
    file_path = test_dir / filename
    
    # Check if file exists, create it if not
    if not file_path.exists():
        print(f"Test file {filename} not found. Creating sample file.")
        if filename == "transcript_sample.txt":
            create_sample_transcript(file_path)
        elif filename == "document_sample.txt":
            create_sample_document(file_path)
        else:
            print(f"No sample generator for {filename}")
            return ""
    
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def create_sample_transcript(file_path):
    """Create a sample transcript file for testing."""
    transcript = """Meeting Transcript: Product Development Team
Date: January 15, 2023
Participants: John Smith, Lisa Wong, Michael Johnson, Sarah Lee

10:00 AM - Meeting Start

John: Good morning everyone. Today we'll be discussing the roadmap for Q2 and addressing some of the issues that came up in our user testing last week.

Lisa: Before we start, I wanted to mention that we got some new feedback from the beta testers. Their main concern is still the loading time for large files.

Michael: I think we need to prioritize that. The current solution isn't scaling well with files over 100MB.

John: I agree. Let's put that at the top of our list.

Sarah: I've been working on an optimization that might help. I can share the preliminary results after this meeting.

10:15 AM - Q2 Roadmap Discussion

John: Let's move on to the Q2 roadmap. The main features we planned were the collaborative editing, export to multiple formats, and the new dashboard design.

Lisa: The design team has completed most of the mockups for the dashboard. I can share those with everyone after the meeting.

Michael: For the collaborative editing feature, we need to decide if we're going with the WebSocket approach or the conflict resolution model.

Sarah: I think the WebSocket approach would be better for real-time collaboration, but it might be more complex to implement.

John: Let's weigh the pros and cons. What's the development timeline difference between the two approaches?

Michael: The WebSocket approach might take an extra 2-3 weeks, but would provide a better user experience.

10:30 AM - Technical Discussion

Sarah: I've been researching some optimization techniques for large file handling. We could implement a streaming parser instead of loading the entire file into memory.

Michael: That's a good idea. We'd need to refactor the file handler class, but it should be doable within our timeline.

Lisa: Would this affect the existing features that rely on the current file handler?

Sarah: We would need to update some dependencies, but I've mapped most of them out already.

John: How confident are we that this will solve the performance issues?

Sarah: Based on my tests with similar implementations, we could see a 70-80% improvement in loading times for large files.

10:45 AM - Action Items

John: Let's summarize the action items. Sarah will continue working on the file optimization and share results by Friday.

Michael: I'll prepare a detailed comparison of the WebSocket vs. conflict resolution approaches by next Monday.

Lisa: I'll share the dashboard designs with everyone today and collect feedback by the end of the week.

John: And I'll update the roadmap to prioritize the file performance improvements. Does anyone have anything else to add?

Sarah: I'll need some help testing the optimization with different file types.

John: Michael, can you help Sarah with that?

Michael: Yes, I can allocate some time on Thursday.

11:00 AM - Meeting Adjourned

John: Thanks everyone. Let's meet again next week to review progress.
"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(transcript)

def create_sample_document(file_path):
    """Create a sample document file for testing."""
    document = """# Project Evaluation Report

## Executive Summary

This document provides an evaluation of the Alpha Project's performance over the past six months. The project has met most of its key milestones but is facing challenges with resource allocation and technical debt that need to be addressed promptly.

## Project Overview

The Alpha Project was initiated in January 2023 with the goal of developing a new customer relationship management system for the sales department. The project was allocated a budget of $500,000 and a timeline of 12 months.

### Key Objectives
- Replace the legacy CRM system
- Integrate with existing marketing automation tools
- Provide mobile access for field sales representatives
- Improve reporting capabilities for management

## Current Status

As of June 2023, the project is 55% complete, which is slightly behind the projected 60% completion target. The following sections provide details on different aspects of the project.

### Technical Progress

The core database architecture has been successfully implemented and the user interface design has been completed for 70% of the planned screens. Integration with two of the four marketing tools has been completed and tested.

However, several technical challenges have emerged:

1. The data migration from the legacy system is proving more complex than anticipated
2. Performance testing has revealed bottlenecks in the reporting module
3. Mobile API development is behind schedule due to unforeseen compatibility issues

### Budget and Resources

The project has currently used 52% of its allocated budget, which is in line with the completion percentage. However, there are concerns about the remaining budget being sufficient due to the following factors:

- Additional development resources needed for mobile API challenges
- Increased complexity of data migration requiring specialized expertise
- Potential infrastructure upgrades needed to address performance issues

### Timeline and Milestones

The project has met the following milestones:
- Database architecture design (completed on time)
- User interface mockups (completed 2 weeks ahead of schedule)
- First integration with marketing tools (completed on time)

The following milestones are at risk:
- Mobile API development (currently 3 weeks behind schedule)
- Data migration planning (2 weeks behind schedule)
- Performance optimization (not started, was scheduled to begin last month)

## Risk Assessment

The project team has identified several key risks that require mitigation:

| Risk | Impact | Probability | Mitigation Strategy |
|------|--------|------------|---------------------|
| Mobile API delays | High | High | Allocate additional resources, consider phased release |
| Data migration issues | High | Medium | Engage data specialists, conduct additional testing |
| Performance bottlenecks | Medium | High | Early optimization, infrastructure upgrades |
| Resource constraints | Medium | Medium | Cross-train team members, adjust timeline |

## Recommendations

Based on the current status and risk assessment, the following recommendations are proposed:

1. **Revise Timeline**: Adjust the project timeline to accommodate the mobile API development challenges, potentially extending by 4-6 weeks.

2. **Resource Allocation**: Add two additional developers with mobile expertise for the next 2 months.

3. **Technical Approach**: Consider a phased implementation approach, prioritizing core functionality for initial release.

4. **Budget Review**: Conduct a comprehensive budget review to ensure sufficient funds for the remaining work.

5. **Knowledge Sharing**: Implement weekly technical deep-dive sessions to address knowledge gaps and improve team collaboration.

## Conclusion

While the Alpha Project is facing some challenges, the core development is progressing well, and the project remains viable. With the recommended adjustments, the project can still deliver significant value to the organization and meet its primary objectives.

## Next Steps

The project manager will:
1. Update the project plan based on these recommendations
2. Meet with stakeholders to review the revised approach
3. Implement changes to resource allocation
4. Schedule weekly status reviews to closely monitor progress

---

Appendix A: Detailed Technical Specifications
Appendix B: Updated Resource Allocation Plan
Appendix C: Risk Management Matrix
"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(document)

def print_chunk_info(chunks):
    """Print information about the chunks for debugging."""
    print(f"Total chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        chunk_type = chunk.get('chunk_type', 'unknown')
        position = chunk.get('position', 'unknown')
        text_preview = chunk['text'][:50].replace('\n', ' ') + "..."
        
        print(f"Chunk {i+1}: {chunk_type}, {position}, {len(chunk['text'])} chars")
        print(f"  Preview: {text_preview}")
    
    print()

def test_transcript_chunking():
    """Test chunking with a transcript document."""
    print("\n--- Testing Transcript Chunking ---")
    
    # Load test transcript
    transcript = load_test_document("transcript_sample.txt")
    print(f"Loaded transcript: {len(transcript)} chars")
    
    # Create chunker
    chunker = DocumentChunker()
    
    # Test with different min_chunks settings
    for min_chunks in [3, 5]:
        print(f"\nTesting with min_chunks={min_chunks}")
        start_time = time.time()
        chunks = chunker.chunk_document(transcript, min_chunks=min_chunks)
        elapsed = time.time() - start_time
        
        print(f"Chunking completed in {elapsed:.3f} seconds")
        print_chunk_info(chunks)
        
        # Check if it recognized as transcript
        is_transcript = any(c.get('chunk_type') == 'transcript_segment' for c in chunks)
        print(f"Recognized as transcript: {is_transcript}")

def test_document_chunking():
    """Test chunking with a structured document."""
    print("\n--- Testing Document Chunking ---")
    
    # Load test document
    document = load_test_document("document_sample.txt")
    print(f"Loaded document: {len(document)} chars")
    
    # Create chunker
    chunker = DocumentChunker()
    
    # Test with different min_chunks settings
    for min_chunks in [3, 5]:
        print(f"\nTesting with min_chunks={min_chunks}")
        start_time = time.time()
        chunks = chunker.chunk_document(document, min_chunks=min_chunks)
        elapsed = time.time() - start_time
        
        print(f"Chunking completed in {elapsed:.3f} seconds")
        print_chunk_info(chunks)
        
        # Check if it recognized as structural
        is_structural = any(c.get('chunk_type') == 'structural_segment' for c in chunks)
        print(f"Used structural chunking: {is_structural}")

def test_large_text_chunking():
    """Test chunking with a large generated text."""
    print("\n--- Testing Large Text Chunking ---")
    
    # Generate a large text
    paragraph = "This is a sample paragraph that will be repeated to create a large document. " * 20
    large_text = paragraph * 50  # Create 50 paragraphs
    
    print(f"Generated large text: {len(large_text)} chars")
    
    # Create chunker
    chunker = DocumentChunker()
    
    # Test with default settings
    print("\nTesting with default settings")
    start_time = time.time()
    chunks = chunker.chunk_document(large_text)
    elapsed = time.time() - start_time
    
    print(f"Chunking completed in {elapsed:.3f} seconds")
    print_chunk_info(chunks)
    
    # Test with specific max_chunk_size
    max_size = 5000
    print(f"\nTesting with max_chunk_size={max_size}")
    start_time = time.time()
    chunks = chunker.chunk_document(large_text, max_chunk_size=max_size)
    elapsed = time.time() - start_time
    
    print(f"Chunking completed in {elapsed:.3f} seconds")
    print_chunk_info(chunks)
    
    # Check average chunk size
    avg_size = sum(len(c['text']) for c in chunks) / len(chunks)
    print(f"Average chunk size: {avg_size:.1f} chars")
    print(f"Largest chunk size: {max(len(c['text']) for c in chunks)} chars")
    print(f"Smallest chunk size: {min(len(c['text']) for c in chunks)} chars")

def run_all_tests():
    """Run all test cases."""
    print("===== Testing DocumentChunkerV2 =====")
    
    test_transcript_chunking()
    test_document_chunking()
    test_large_text_chunking()
    
    print("\n===== All tests completed =====")

if __name__ == "__main__":
    run_all_tests()