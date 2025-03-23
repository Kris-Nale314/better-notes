# 🧪 IssuesCrew: Assembling a Specialized Multi-Agent Team

> A modular, scalable, and LLM-optimized crew for identifying problems, risks, and challenges in complex documents—built on macro-chunks and powered by meta-agent planning.

---

## 🧭 What Is the IssuesCrew?

`IssuesCrew` is a **specialized agentic pipeline** designed to extract and analyze issues from long-form documents like meeting transcripts, policy papers, or technical reports.

It is not a monolithic function—it is a **configurable team** of agents, each with a unique role, working in a coordinated sequence to generate high-quality insights.

---

## ⚙️ How It Works: Overview

```mermaid
flowchart LR
    A[Document Upload] --> B[Macro-Chunking (10k tokens)]
    B --> C[Planner Agent]
    C --> D1[Extractor Agent]
    D1 --> D2[Aggregator Agent]
    D2 --> D3[Evaluator Agent]
    D3 --> D4[Formatter Agent]
    D4 --> E[🧪 Reviewer Agent (Optional)]
    E --> F[Final Report + Metadata + Chat]

    style C fill:#dfefff,stroke:#000,stroke-width:1px
    style E fill:#fdf2ff,stroke:#000,stroke-width:1px
```

---

## 🧠 Agent Roles (Generated Dynamically)

The crew is **assembled dynamically** based on the `issues_config.json` file and customized in real time using the **Planner Agent** (formerly Instructor). This meta-agent tailors each agent’s:

- 🎯 **Goal**
- 🧾 **Instructions**
- 🧲 **Emphasis**
- 📦 **Output format**

Each agent is a subclass of `BaseAgent` and uses a shared LLM backend for consistency.

---

## 🛠️ Agent Workflow

| Agent Type     | Role | Description |
|----------------|------|-------------|
| 🧭 **PlannerAgent** | Meta-Agent | Analyzes the document metadata and user preferences to generate customized instructions for each agent |
| 🕵️‍♂️ **ExtractorAgent** | Specialized Agent | Parses macro-chunks in parallel (using async) to detect potential issues |
| 🧩 **AggregatorAgent** | Specialized Agent | Deduplicates, clusters, and consolidates issues across all chunks |
| 🧮 **EvaluatorAgent** | Specialized Agent | Assigns severity and impact levels using rules and custom criteria |
| 🧾 **FormatterAgent** | Specialized Agent | Transforms structured data into a user-friendly, HTML-formatted report |
| 🧪 **ReviewerAgent** *(optional)* | Specialized Agent | Provides a QA layer with scoring (clarity, completeness, etc.) and feedback summary |

---

## 🧱 How Agents Are Instantiated

```python
self.extractor_agent = ExtractorAgent(llm_client, crew_type="issues", config=config)
self.aggregator_agent = AggregatorAgent(llm_client, crew_type="issues", config=config)
# etc.
```

All agents:
- Inherit from `BaseAgent`
- Are configured via `issues_config.json`
- Can override goals, templates, rate limits, verbosity, etc.

---

## 🧬 How the Crew Is Built

When `process_document()` is called:

1. **Document Preparation**  
   The raw text is macro-chunked using custom logic (~10k tokens per chunk).

2. **Planner Generates Instructions**  
   Tailored prompts are generated per agent using metadata + user inputs.

3. **Extractor Runs in Parallel (Async)**  
   Each macro-chunk is analyzed independently.

4. **Sequential Workflow**  
   Aggregation → Evaluation → Formatting → (Optional) Review

5. **Metadata + Output Assembly**  
   Results are returned with embedded metadata for display, download, or chat.

---

## 🧠 Meta-Agent-Led Config Adaptation

Example of what the Planner generates:

```json
{
  "evaluation": {
    "instructions": "Assign severity scores (Low/Med/High) to each issue based on organizational impact.",
    "emphasis": "Prioritize cross-cutting issues and unresolved risks."
  }
}
```

This is injected into the agent pipeline dynamically before execution.

---

## 🧾 Output Example

The final output includes:

- ✅ **Formatted HTML report**
- 🧪 **Review scores (if reviewer is enabled)**
- 📊 **Agent logs + execution stats**
- 💬 **Document-aware chat support**
- 📁 **Downloadable export**

---

## 🧰 Why This Approach Is Valuable

| Benefit | Why It Matters |
|---------|----------------|
| ✅ **Modular** | Swap out agents or crew types (e.g., `actions_crew`) with no structural changes |
| 📊 **Auditable** | Tracks execution time, input/output lengths, errors per agent |
| 🔄 **Reusable** | Framework supports new assessments (actions, risks, opportunities) |
| 🧠 **Explainable** | Instructions, metadata, and output structure are transparent |
| 🧱 **Scalable** | Handles long-form documents via macro-chunking and async parallelism |
| 🧪 **Refinable** | Supports post-analysis chat and reprocessing with updated inputs |

---

## 🔮 What’s Next

- ✨ Swappable Planner/Instructor architecture for different planning strategies
- 🧪 Comparison across crews (e.g., Issue Crew vs. Risk Crew)
- 📉 Use of retrieval + memory for multi-session agent interactions
- 🔧 Live agent tuning from frontend UI (agent config editor)

---

## 📁 Directory Structure

```
crews/
├── issues_crew.py        # This file (IssuesCrew)
├── action_crew.py        # Other crew types
agents/
├── extractor.py          # Agent implementations
├── base.py               # BaseAgent
├── config/
│   ├── issues_config.json
```

---

## 🧠 In Summary

The `IssuesCrew` represents a **domain-specific multi-agent system** orchestrated by a meta-agent and powered by configuration, parallel processing, and task specialization.

It transforms long documents into actionable, trustworthy insight—and serves as a blueprint for building reusable analytical pipelines across domains.

---