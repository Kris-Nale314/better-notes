# Assessment Crew Framework

<div align="center">
  <img src="https://raw.githubusercontent.com/kris-nale314/better-notes/main/docs/images/logo.svg" alt="Better-Notes logo" width="120px"/>
  <h3>Specialized Agent Teams for Document Analysis</h3>
</div>

## Introduction

The Assessment Crew Framework is the core innovation that powers Better Notes' analytical capabilities. This modular, multi-agent system organizes specialized AI agents into configurable "crews" - teams that collaborate to perform specific types of document analysis. This document explains the crew framework, how agents interact, and how the system can be extended with new capabilities.

## What is an Assessment Crew?

An Assessment Crew is a team of specialized AI agents that work together through a structured pipeline to analyze documents for specific insights. Each crew:

- Has a defined analytical purpose (e.g., identifying issues, extracting action items)
- Contains specialized agents with specific roles
- Follows a configurable workflow with defined stages
- Uses a shared configuration for consistency
- Produces structured, metadata-rich results

## The Agent Pipeline

All crews in Better Notes follow a consistent agent pipeline:

<div align="center">
<img src="https://via.placeholder.com/800x200?text=Agent+Pipeline+Diagram" alt="Agent Pipeline" width="90%"/>
</div>

### 1. Planner Agent

The Planner acts as a meta-agent that orchestrates the entire process:

- Analyzes the document structure and content
- Determines optimal processing strategies
- Creates tailored instructions for each specialist agent
- Adapts to document type and user preferences

Rather than using fixed prompts, the Planner generates document-specific guidance for each agent, enabling much more targeted analysis.

### 2. Extractor Agent

The Extractor identifies relevant information from document chunks:

- Processes each document chunk independently
- Identifies items relevant to the crew's focus (issues, actions, etc.)
- Adds initial metadata (location, keywords, etc.)
- Operates in parallel across chunks for efficiency

The Extractor embodies the principle of comprehensive coverage, ensuring nothing important is missed.

### 3. Aggregator Agent

The Aggregator combines and refines the Extractor's findings:

- Consolidates similar items from different chunks
- Eliminates duplicates while preserving unique aspects
- Enhances metadata (mention frequency, confidence, etc.)
- Resolves conflicts and inconsistencies

This aggregation step is critical for transforming fragmented observations into coherent insights.

### 4. Evaluator Agent

The Evaluator assesses importance and impact of the aggregated items:

- Applies assessment criteria (severity, priority, etc.)
- Provides rationales for assessments
- Adds impact analysis and implications
- Identifies relationships between items

The Evaluator performs the critical thinking that elevates raw findings to evaluated insights.

### 5. Formatter Agent

The Formatter creates structured, navigable reports:

- Organizes content by importance and category
- Applies appropriate templates for the analysis type
- Creates executive summaries and overview sections
- Enhances readability with visual elements

This step transforms analytical content into a user-friendly presentation.

### 6. Reviewer Agent

The Reviewer performs quality control on the final output:

- Ensures analysis aligns with user requirements
- Checks for consistency and completeness
- Evaluates quality across multiple dimensions
- Provides improvement suggestions

This final quality check ensures the output meets high standards before delivery.

## Metadata Layering

A key innovation in the Assessment Crew Framework is **progressive metadata enhancement**. Each agent in the pipeline adds and refines metadata, creating increasingly rich context:

<table>
<tr>
  <th>Agent</th>
  <th>Original Metadata</th>
  <th>Enhanced Metadata</th>
</tr>
<tr>
  <td>Extractor</td>
  <td>Document text</td>
  <td>+ Item identification, initial categorization, location context, source chunk</td>
</tr>
<tr>
  <td>Aggregator</td>
  <td>Extracted items</td>
  <td>+ Mention frequency, confidence scores, source chunks, variation tracking</td>
</tr>
<tr>
  <td>Evaluator</td>
  <td>Aggregated items</td>
  <td>+ Severity/priority ratings, rationales, impact assessments, relationships</td>
</tr>
<tr>
  <td>Formatter</td>
  <td>Evaluated items</td>
  <td>+ Organizational structure, visual prioritization, navigation elements</td>
</tr>
<tr>
  <td>Reviewer</td>
  <td>Formatted report</td>
  <td>+ Quality scores, improvement suggestions, overall assessment</td>
</tr>
</table>

This layered approach allows each agent to build on the work of previous agents, creating progressively richer analysis.

## Configuration-Driven Architecture

Assessment Crews are configured through JSON files that define all aspects of their behavior:

```json
{
  "analysis_definition": {
    "issue": {
      "definition": "Any problem, challenge, risk, or concern that may impact objectives",
      "examples": ["Missing requirements", "Technical limitations", "Process inefficiencies"],
      "non_examples": ["Simple observations without negative impact"]
    },
    "severity_levels": {
      "critical": "Immediate threat to operations, security, or compliance",
      "high": "Significant impact on effectiveness or efficiency",
      "medium": "Causes ongoing inefficiency or limitations",
      "low": "Minor inconvenience or concern"
    }
  },
  
  "agents": {
    "extractor": {
      "role": "Issue Extractor",
      "goal": "Identify all potential issues in document chunks",
      "instructions": "Analyze the document chunk to identify issues...",
      "output_format": {
        "issues": [
          {
            "title": "Issue title",
            "description": "Detailed description",
            "initial_severity": "One of: critical, high, medium, low",
            "keywords": ["relevant", "keywords"],
            "location_context": "Section reference"
          }
        ]
      }
    },
    // Configuration for other agents...
  }
}
```

This configuration-driven approach enables:

1. **Easy Customization**: Crews can be modified without code changes
2. **Consistent Behavior**: All agents follow the same structural principles
3. **Clear Expectations**: Output formats are explicitly defined
4. **Shared Understanding**: Analysis definitions are consistent across agents

## Implemented Crew Types

Better Notes currently implements the following crew types:

### Issues Crew

The Issues Crew identifies problems, challenges, risks, and concerns in documents:

- **Focus**: Potential negative impacts and areas needing attention
- **Assessment Criteria**: Severity based on impact and urgency
- **Organization**: Grouped by severity (critical, high, medium, low)
- **Unique Features**: Impact assessment, relationship mapping

### Action Items Crew

The Action Items Crew extracts tasks, commitments, and follow-up requirements:

- **Focus**: What needs to be done, by whom, and when
- **Assessment Criteria**: Priority based on urgency and importance
- **Organization**: Grouped by owner and timeline
- **Unique Features**: Ownership attribution, due date extraction

### Insights Crew

The Insights Crew discovers key themes and notable information:

- **Focus**: Important observations, patterns, and noteworthy content
- **Assessment Criteria**: Relevance and significance
- **Organization**: Grouped by theme and type
- **Unique Features**: Quote extraction, theme identification

## Error Handling and Resilience

Assessment Crews include robust error handling at every stage:

1. **Graceful Degradation**: If an agent encounters an error, the system attempts to continue with partial results
2. **Error Logging**: Detailed error information is captured for debugging
3. **Fallback Strategies**: Alternative processing paths when primary approaches fail
4. **Status Tracking**: Comprehensive tracking of process state enables recovery

This resilience ensures that analysis can complete even when unexpected issues occur.

## Parallel Processing

Assessment Crews optimize performance through intelligent parallel processing:

- **Chunk-Level Parallelism**: Multiple document chunks are processed simultaneously
- **Rate Limiting**: API request rates are managed to avoid rate limiting
- **Prioritized Processing**: Critical sections can be prioritized
- **Adaptive Concurrency**: Processing adapts based on document size and complexity

This approach balances performance and resource utilization.

## Extending with New Crews

The Assessment Crew Framework is designed for extensibility. New crews can be added by:

1. Creating a configuration file defining the analysis type
2. Implementing a crew class that orchestrates the agent pipeline
3. Adding UI elements for the new analysis type

New crews inherit all the capabilities of the framework while specializing in their specific analysis focus.

## Implementation Details

The Assessment Crew Framework is implemented through several key components:

- `BaseAgent` class: Provides foundation functionality for all agents
- Specialized agent classes: Implement specific agent roles
- Crew classes: Orchestrate the agent pipeline for specific analysis types
- Configuration loader: Parses and validates JSON configurations
- Progress tracking: Monitors and reports on process status

## Technical Implementation Example

Here's how the IssuesCrew processes a document:

```python
# Initialize the crew with appropriate configuration
issues_crew = IssuesCrew(
    llm_client=llm_client,
    config_path="agents/config/issues_config.json",
    verbose=True
)

# Process the document
result = issues_crew.process_document(
    document_text=document_text,
    document_info=document_info,
    user_preferences={
        "detail_level": "standard",
        "focus_areas": ["technical", "process"]
    },
    progress_callback=update_progress
)

# Result contains structured analysis with metadata at each level
```

## Future Directions

The Assessment Crew Framework is being enhanced with several planned improvements:

1. **Multi-Document Crews**: Analyzing sets of related documents
2. **Interactive Crews**: Agents that can request clarification during analysis
3. **Customizable Pipelines**: User-configurable agent workflows
4. **Domain-Specific Crews**: Specialized crews for specific industries or document types

## Conclusion

The Assessment Crew Framework represents a sophisticated approach to document analysis that moves beyond simple AI applications toward intelligent, collaborative systems. By combining specialized agents, progressive metadata enhancement, and configurable workflows, it delivers insights that are more comprehensive, better organized, and more actionable than traditional approaches.

This architecture demonstrates how multi-agent AI systems can tackle complex analytical tasks in ways that more closely resemble human expert teams than simple automation.