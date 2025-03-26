# 🔄 `better-notes`: Workflow

<div align="center">
  <img src="https://raw.githubusercontent.com/kris-nale314/better-notes/main/docs/images/logo.svg" alt="Better-Notes logo" width="120px"/>
  <h3>How AI Agents Transform Documents into Insights</h3>
  <p><em>A detailed walkthrough of the multi-agent architecture in action</em></p>
</div>

This document describes the complete workflow of the `better-notes` system, from document submission to interactive exploration. It illustrates how multiple AI agents collaborate to transform unstructured documents into structured, actionable insights.

## 1️⃣ Input & Configuration Phase

The journey begins with document submission and configuration:

- **Document Submission**: User uploads or pastes document text (meeting transcripts, reports, articles, etc.)
- **Analysis Type Selection**: User selects the type of analysis:
  * 🚨 **Issues Analysis**: Identify problems, challenges, and risks
  * ✅ **Action Items**: Extract tasks, commitments, and assignments
  * 💡 **Insights**: Discover key themes and notable information

- **Parameter Configuration**:
  * 📊 **Detail Level**: Essential (high-priority only), Standard (balanced), or Comprehensive (in-depth)
  * 🎯 **Focus Areas**: Domain-specific categories relevant to the document
  * 📝 **Custom Instructions**: Optional specialized guidance for the analysis

- **Model Selection**: Choose appropriate LLM based on needs:
  * 💪 **GPT-4**: For complex reasoning tasks (Planning, Review)
  * ⚡ **GPT-3.5 Turbo**: For efficient extraction and processing
  * 🔄 **Model Flexibility**: System designed to work with different models

- **Configuration Loading**: System loads corresponding JSON configuration:
```json
{
  "crew_type": "issues",
  "issue_definition": {
    "description": "Any problem, challenge, risk, or concern...",
    "severity_levels": {
      "critical": "Immediate threat requiring urgent attention",
      "high": "Significant impact requiring prompt attention",
      "medium": "Moderate impact that should be addressed",
      "low": "Minor impact with limited consequences"
    },
    "categories": [
      "technical", "process", "resource", "quality", "risk", "compliance"
    ]
  },
  "workflow": {
    "enabled_stages": ["document_analysis", "chunking", "planning", 
                       "extraction", "aggregation", "evaluation", 
                       "formatting", "review"],
    "agent_roles": {
      // Agent role definitions...
    }
  }
}
```

## 2️⃣ System Initialization Phase

Once configured, the system initializes its components:

- **Orchestrator Creation**: `OrchestratorFactory` creates an `Orchestrator` with selected parameters:
```python
orchestrator = OrchestratorFactory.create_orchestrator(
    api_key=api_key,
    model=selected_model,
    temperature=temperature,
    max_chunk_size=max_chunk_size,
    max_rpm=max_rpm,
    config_manager=config_manager
)
```

- **Processing Context Creation**: System creates a context object to maintain state:
```python
context = ProcessingContext(document_text, options)
```

- **Crew Assembly**: The appropriate crew is instantiated based on analysis type:
```python
crew = self._get_crew(crew_type)  # e.g., IssuesCrew, ActionsCrew
```

- **Document Pre-processing**: System performs initial document analysis:
  * 📄 Document type detection (transcript, report, article)
  * 📊 Basic statistics (word count, estimated tokens)
  * 🔍 Preview analysis for planning context

- **Agent Initialization**: Required agents are prepared with appropriate configurations:
```python
planner = self._get_agent("planner")
extractor = self._get_agent("extractor")
# ... initialize other agents
```

## 3️⃣ Planning & Instruction Phase

The Planner agent creates a tailored approach for the document:

- **Document Analysis**: Planner examines document structure and content:
  * 📑 Identifies document sections and organization
  * 🔍 Determines document type-specific characteristics
  * 🧮 Assesses complexity and scope

- **User Preference Integration**: Planner incorporates user-selected parameters:
  * 🎯 Adjusts focus based on selected areas
  * 📏 Calibrates depth based on detail level
  * 📝 Integrates custom instructions

- **Dynamic Instruction Generation**: Planner creates tailored instructions for each agent:
```python
# Planning result example
{
  "extractor": {
    "instructions": "Identify issues from each document chunk, focusing on technical risks and resource constraints. Assign severity based on potential impact.",
    "emphasis": "Pay special attention to statements about timeline feasibility and budget limitations."
  },
  "aggregator": {
    "instructions": "Combine similar issues while preserving important context from different document sections.",
    "emphasis": "Ensure technical details aren't lost during consolidation."
  },
  // Instructions for other agents...
}
```

- **Pipeline Configuration**: Processing stages are sequenced based on document complexity:
  * 🧩 Number of chunks determined by document length
  * ⚙️ Parallelization strategy established
  * 📋 Progress tracking initialized

## 4️⃣ Processing Pipeline Phase

The document flows through specialized processing stages:

### 🔍 Extraction Stage

The Extractor agent processes each chunk to identify relevant items:

```python
async def _extract_issues(self, context):
    # Get the extractor agent
    extractor = self._get_agent("extractor")
    
    # Extract issues from all chunks in parallel
    extraction_results = await extractor.process(context)
    return extraction_results
```

- Adds initial metadata (location, context, keywords)
- Works in parallel across document chunks
- Assigns preliminary classifications and severity

**Example Extractor Output (per chunk):**
```json
{
  "chunk_id": 2,
  "issues": [
    {
      "title": "Unclear project scope",
      "description": "The project scope is ambiguously defined, with multiple stakeholders expressing different expectations.",
      "severity": "high",
      "category": "process",
      "context": "In the meeting section where marketing and engineering discussed requirements."
    },
    // More extracted issues...
  ]
}
```

### 🧩 Aggregation Stage

The Aggregator agent combines related items from different chunks:

```python
async def _aggregate_issues(self, context):
    # Get the aggregator agent
    aggregator = self._get_agent("aggregator")
    
    # Aggregate issues from all extraction results
    aggregated_result = await aggregator.process(context)
    return aggregated_result
```

- Eliminates duplicates while preserving important variations
- Enhances metadata with frequency and source information
- Creates consolidated list of unique items

**Example Aggregator Output:**
```json
{
  "aggregated_issues": [
    {
      "title": "Unclear project scope",
      "description": "The project scope is ambiguously defined, with multiple stakeholders expressing different expectations.",
      "severity": "high",
      "category": "process",
      "mention_count": 3,
      "source_chunks": [2, 4, 7],
      "contexts": [
        "In the meeting section where marketing and engineering discussed requirements.",
        "During the stakeholder review section.",
        "In the risk assessment portion."
      ],
      "variations": ["scope ambiguity", "requirement uncertainty"]
    },
    // More aggregated issues...
  ]
}
```

### ⚖️ Evaluation Stage

The Evaluator agent assesses importance and impact:

```python
async def _evaluate_issues(self, context):
    # Get the evaluator agent
    evaluator = self._get_agent("evaluator")
    
    # Evaluate aggregated issues
    evaluated_result = await evaluator.process(context)
    return evaluated_result
```

- Assigns final severity/priority ratings
- Adds rationales for assessments
- Creates relationship mappings between items
- Generates executive summary of findings

**Example Evaluator Output:**
```json
{
  "executive_summary": "The document reveals several critical issues centered around project scope, resource constraints, and technical debt. The most severe concerns relate to timeline feasibility and budget limitations.",
  
  "critical_issues": [
    {
      "title": "Unclear project scope",
      "description": "The project scope is ambiguously defined, with multiple stakeholders expressing different expectations.",
      "severity": "critical",  // Upgraded from high based on impact analysis
      "category": "process",
      "impact": "High risk of project failure, budget overruns, and delayed delivery.",
      "rationale": "Scope ambiguity affects all aspects of planning and execution.",
      "related_issues": ["Timeline feasibility", "Resource allocation"]
    },
    // More critical issues...
  ],
  
  "high_issues": [
    // High-priority issues...
  ],
  
  // Medium and low issues...
}
```

### 📊 Formatting Stage

The Formatter agent creates structured output:

```python
async def _format_report(self, context):
    # Get the formatter agent
    formatter = self._get_agent("formatter")
    
    # Format the evaluated results into a report
    formatted_result = await formatter.process(context)
    return formatted_result
```

- Organizes content by priority/category
- Applies appropriate visual formatting
- Creates navigable HTML report structure

**Example Formatter Output:**
```html
<div class="issues-report">
  <h1>Issues Analysis Report</h1>
  
  <div class="executive-summary">
    <h2>📋 Executive Summary</h2>
    <p>The document reveals several critical issues centered around project scope, resource constraints, and technical debt...</p>
  </div>
  
  <div class="issues-section">
    <h2>🔴 Critical Issues (3)</h2>
    <div class="issue critical">
      <h3>Unclear project scope</h3>
      <div class="issue-metadata">
        <span class="category">Process</span>
        <span class="severity critical">Critical</span>
      </div>
      <p>The project scope is ambiguously defined, with multiple stakeholders expressing different expectations.</p>
      <div class="impact">
        <strong>Impact:</strong> High risk of project failure, budget overruns, and delayed delivery.
      </div>
    </div>
    <!-- More issues... -->
  </div>
  
  <!-- Other severity sections... -->
</div>
```

### 🔎 Review Stage

The Reviewer agent performs quality assessment:

```python
async def _review_report(self, context):
    # Get the reviewer agent
    reviewer = self._get_agent("reviewer")
    
    # Review the formatted report
    review_result = await reviewer.process(context)
    return review_result
```

- Verifies alignment with user requirements
- Provides feedback on analysis quality
- Suggests improvements for future analyses

**Example Reviewer Output:**
```json
{
  "summary": "The analysis provides comprehensive coverage of the document's issues with appropriate severity assessments.",
  "assessment": {
    "completeness_score": 4.5,
    "accuracy_score": 4.2,
    "organization_score": 4.8,
    "alignment_score": 4.0
  },
  "improvement_suggestions": [
    {
      "area": "Technical details",
      "suggestion": "Consider providing more specific context for technical issues."
    },
    {
      "area": "Related issues",
      "suggestion": "The relationships between budget and timeline issues could be more clearly articulated."
    }
  ]
}
```

## 5️⃣ Results Presentation Phase

The system presents the completed analysis to the user:

- **Report Display**: Formatted HTML report is presented in the UI
- **Quality Metrics**: Assessment scores from reviewer are displayed
- **Execution Statistics**: Processing time and agent metrics are shown
- **Download Options**: User can save the report in various formats
- **Technical Details**: Access to processing metadata is provided

## 6️⃣ Interactive Exploration Phase

The user can now interactively explore the document:

- **Document Chat**: Enables conversation with document using analysis context:
  ```python
  def display_chat_interface(llm_client, document_text, summary_text, document_info):
      # Initialize chat state
      if "chat_history" not in st.session_state:
          st.session_state.chat_history = []
      
      # Chat input and processing
      user_question = st.text_input("Your question:")
      if st.button("Send") and user_question:
          process_chat_question(llm_client, user_question, document_text, summary_text)
  ```

- **Analysis Refinement**: Allows modifications to analysis parameters:
  * Adjust detail level for more or less depth
  * Change focus areas to highlight different aspects
  * Add specific instructions for targeted analysis

- **Comparative Analysis**: Run multiple analyses with different parameters:
  * Compare results across different focus areas
  * Identify consistent vs. variable findings

- **Insight Development**: Iteratively refine understanding through interaction:
  * Build on initial findings with targeted questions
  * Explore implications of identified issues
  * Develop action plans based on analysis

## 🔄 The Unified Workflow

This comprehensive workflow demonstrates how `better-notes` transforms documents through a coordinated system of AI agents, each with specialized expertise:

<div align="center">
  <img src="https://raw.githubusercontent.com/kris-nale314/better-notes/main/docs/images/workflow.svg" alt="better-notes flow" width="90%"/>
  <p><em>From raw document to structured insights through collaborative AI agents</em></p>
</div>

What makes this approach powerful is:

1. **Specialization**: Each agent focuses deeply on one aspect of analysis
2. **Progressive Enhancement**: Information is enriched at each processing stage
3. **Adaptable Architecture**: Configuration-driven behavior without code changes
4. **Interactive Exploration**: Analysis becomes a starting point for deeper understanding

This multi-agent architecture creates a more comprehensive, organized, and nuanced analysis than would be possible with a single model - transforming the way we extract meaning from complex documents.