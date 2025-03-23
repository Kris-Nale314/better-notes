# 🤖 Assessment Crews: Collaborative AI Teams

## What Are Assessment Crews?

Assessment Crews are specialized teams of AI agents that work together to analyze documents through a structured, configurable workflow. Each crew is designed for a specific analysis type (issues, actions, opportunities, etc.) but follows the same meta-pattern of collaborative intelligence.

Think of a crew as a team of specialists with complementary skills working on your document with a shared goal.

## The Crew Workflow

Every crew follows a consistent pattern:

1. **Document Preparation**
   - Split document into macro-chunks (10k tokens)
   - Analyze metadata (document type, length, structure)

2. **Planning**
   - Planner Agent creates document-specific instructions
   - User preferences shape the analysis approach

3. **Parallel Extraction**
   - Extractor Agent processes each chunk concurrently
   - First-pass identification of relevant content

4. **Sequential Refinement**
   - Aggregator Agent combines and deduplicates findings
   - Evaluator Agent assesses importance and priority
   - Formatter Agent creates structured reports
   - Reviewer Agent (optional) performs quality check

5. **Final Output**
   - HTML report with embedded metadata
   - Support for post-analysis chat

## JSON Configuration

Each crew is defined by a JSON configuration file (e.g., `issues_config.json`) that controls the behavior of all agents:

```json
{
  "metadata": {
    "version": "2.0",
    "description": "Issues identification configuration"
  },
  
  "analysis_definition": {
    "issue": {
      "definition": "Any problem, challenge, risk, or concern",
      "examples": ["Missing requirements", "Technical limitations"]
    },
    "severity_levels": {
      "critical": "Immediate threat to operations",
      "high": "Significant impact on effectiveness",
      "medium": "Causes ongoing inefficiency",
      "low": "Minor inconvenience or concern"
    }
  },
  
  "agents": {
    "extraction": {
      "role": "Issue Extractor",
      "goal": "Identify all potential issues in document chunks",
      "instructions": "Analyze the document chunk to identify issues...",
      "output_format": {...}
    },
    // Other agent definitions
  }
}
```

## Crew Types

The Better Notes system supports (or will support) multiple assessment crew types:

1. **Issues Crew**
   - Identifies problems, risks, and challenges
   - Categorizes by severity and impact
   - Focuses on negative aspects requiring attention

2. **Actions Crew** (Planned)
   - Extracts tasks, assignments, and commitments
   - Tracks ownership, deadlines, and dependencies
   - Focuses on actionable next steps

3. **Insights Crew** (Planned)
   - Analyzes key themes, decisions, and context
   - Identifies significant statements and observations
   - Focuses on understanding the document's meaning

Each crew uses the same agent architecture but with different configurations, prompt strategies, and output formats.

## Implementation Details

Crews are implemented as Python classes that:
- Initialize with a shared LLM client and configuration
- Contain instances of all specialized agents
- Manage the workflow from document to final output
- Handle parallel processing and error recovery
- Track execution statistics and metadata

```python
issues_crew = IssuesCrew(
    llm_client=llm_client,
    config_path="agents/config/issues_config.json",
    verbose=True
)

result = issues_crew.process_document(
    document_text,
    document_info=document_info,
    user_preferences=user_preferences
)
```

## Data Flow and Metadata Layering

A key aspect of crews is how they layer metadata at each stage:

1. **Extraction Layer**
   - Chunk-specific information
   - Initial classification attempts
   - Raw content identifiers

2. **Aggregation Layer**
   - Cross-reference tracking
   - Duplicate elimination
   - Mention frequency

3. **Evaluation Layer**
   - Importance assessments
   - Priority scores
   - Relationship mapping

4. **Formatting Layer**
   - Structural organization
   - Visual prioritization
   - Navigation aids

This metadata layering creates a rich, traceable analysis that gets smarter at each step.

## The Power of Modular Design

The crew architecture enables several powerful capabilities:

1. **Configurability**
   - Change behavior through JSON without code changes
   - Define new assessment types with configuration alone

2. **Parallel Processing**
   - Scale to very large documents (100k+ tokens)
   - Process chunks concurrently for faster results

3. **Specialization with Coordination**
   - Each agent excels at a specific task
   - Complex analysis emerges from simple components

4. **Flexible Deployment**
   - Use as an API service
   - Embed in applications
   - Run as a standalone tool

## Summary

Assessment Crews represent a new approach to document analysis that combines the best aspects of:
- Specialized AI agents
- Configurable workflows
- Metadata-enhanced processing
- Parallel and sequential operations

This architecture enables Better Notes to perform complex document analysis that adapts to both document content and user needs, creating insights that would be difficult to achieve with any single model approach.