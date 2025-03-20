"""
Test script for pass processing functionality.
"""

import os
import asyncio
import json
from dotenv import load_dotenv
from lean.options import ProcessingOptions
from lean.async_openai_adapter import AsyncOpenAIAdapter
from passes.passes import create_pass_processor

# Load environment variables
load_dotenv()

async def test_issue_identification():
    """Test issue identification pass."""
    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        return
    
    # Create LLM client
    llm_client = AsyncOpenAIAdapter(
        model="gpt-3.5-turbo",
        api_key=api_key,
        temperature=0.2
    )
    
    # Create options
    options = ProcessingOptions(
        model_name="gpt-3.5-turbo",
        temperature=0.2,
        passes=["issue_identification"],
        pass_options={
            "issue_identification": {
                "severity_threshold": "medium"
            }
        }
    )
    
    # Create pass processor
    processor = create_pass_processor("issue_identification", llm_client, options)
    if not processor:
        print("ERROR: Failed to create issue identification pass processor")
        return
    
    # Sample document text
    document_text = """
    Meeting Transcript: Project Falcon Sprint Review
    Date: March 15, 2023
    
    Participants: John (Product Manager), Sarah (Lead Developer), Mike (QA Lead), Lisa (UX Designer)
    
    John: Welcome everyone to our sprint review for Project Falcon. Let's go through what we've accomplished and the challenges we're facing.
    
    Sarah: We've made good progress on the API integration, but we're still having issues with the authentication flow. The third-party service keeps timing out randomly, and it's affecting our ability to complete the user profile feature.
    
    Mike: That's right. In testing, we're seeing about 15% of authentication attempts fail because of this timeout issue. It's a critical problem because users can't create accounts reliably.
    
    John: That's concerning. Is there a workaround we can implement?
    
    Sarah: We could cache credentials and retry automatically, but that introduces security concerns. We'd need proper approval from the security team before implementing that.
    
    Lisa: From the UX perspective, we've completed the new dashboard design, but I'm worried about the mobile responsiveness. The current layout breaks on smaller screens, and about 40% of our users are on mobile devices.
    
    John: Let's prioritize fixing that. Can we get it done this week?
    
    Lisa: Honestly, it requires a significant redesign of the component structure. It would take at least two weeks to do it properly.
    
    Mike: Another issue is the performance on the reporting page. It's taking more than 8 seconds to load on average, which is way above our 3-second target.
    
    Sarah: That's because we're loading all the data at once. We need to implement pagination or progressive loading.
    
    John: OK, let's summarize. We have three main issues: authentication timeouts, mobile responsiveness, and report loading performance. Anything else?
    
    Mike: We also have some minor bugs in the notification system. Sometimes notifications don't clear properly after being read. It's not critical but it's annoying for users.
    
    Sarah: And we're still about 20% over budget on AWS costs because we haven't optimized the database queries yet.
    
    John: Alright, let's prioritize these for the next sprint. The authentication issue seems most critical, followed by the mobile responsiveness problem.
    """
    
    # Sample document info
    document_info = {
        "is_meeting_transcript": True,
        "preview_analysis": {
            "client_name": "Project Falcon",
            "meeting_purpose": "Sprint Review",
            "key_topics": ["Development", "API Integration", "UX Design", "Performance"]
        }
    }
    
    # Create a simple progress callback
    def progress_callback(progress, message):
        print(f"Progress: {progress:.0%} - {message}")
    
    # Process document
    print("\nProcessing document with issue identification pass...")
    result = await processor.process_document(
        document_text=document_text,
        document_info=document_info,
        progress_callback=progress_callback
    )
    
    # Print results
    print("\nIssue Identification Results:")
    if 'result' in result:
        issues_result = result['result']
        
        if 'issues' in issues_result:
            issues = issues_result['issues']
            print(f"Found {len(issues)} issues:")
            
            for i, issue in enumerate(issues):
                print(f"\n{i+1}. {issue.get('title', 'Untitled')} ({issue.get('severity', 'unknown').upper()})")
                print(f"   Description: {issue.get('description', 'No description')}")
                if 'speaker' in issue and issue['speaker']:
                    print(f"   Mentioned by: {issue['speaker']}")
                if 'context' in issue and issue['context']:
                    print(f"   Context: \"{issue['context']}\"")
        else:
            print("No issues found.")
            
        if 'summary' in issues_result:
            print(f"\nSummary: {issues_result['summary']}")
    else:
        print("Error: Unexpected result format")
        print(json.dumps(result, indent=2))

# Run the test
if __name__ == "__main__":
    asyncio.run(test_issue_identification())