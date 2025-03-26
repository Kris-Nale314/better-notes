<div align="center">
  <img src="https://raw.githubusercontent.com/kris-nale314/better-notes/main/docs/images/logo.svg" alt="better-notes logo" width="180px"/>
  <h3>Transform documents into structured insights with collaborative AI agents</h3>
</div>

# better-notes
### *Agentic Document Analysis for Meaningful Insights*
---

> *"Imagine if a team of analysts could review every document you read, extracting exactly what matters to you. That's the idea behind Better Notes."*

## What is `better-notes`?

`better-notes` is a project that uses AI agents working together to analyze documents and extract meaningful insights. It started as a personal exploration into document analysis and evolved into a playground for testing multi-agent AI architectures.

`better-notes` breaks analysis into specialized tasks assigned to different agents, much like how a human team might divide complex work.

<div align="center">
  <img src="https://raw.githubusercontent.com/kris-nale314/better-notes/main/docs/images/logic.svg" alt="better-notes Logic" width="90%"/>
</div>

## 💡 Key Features

* **Issues Analysis** - Identify problems, challenges, and risks in documents, organized by severity
* **Action Items** - Extract tasks, commitments, and assignments with clear ownership
* **Smart Chunking** - Handle long documents by processing in 10k-token macro-chunks
* **Post-Analysis Chat** - Have conversations with your documents after analysis
* **Progressive Enhancement** - Each agent adds layers of metadata and insight

## 🧠 What is an Agent?

In `better-notes`, an "agent" isn't just a fancy name for a function call. Each agent:

* Has a **specialized role** with specific responsibilities
* Follows **dynamically generated instructions** tailored to each document
* Maintains **state and context** throughout the analysis process
* Makes **independent decisions** within its domain of expertise

Think of agents as specialized experts who focus deeply on one aspect of document analysis, combining their insights to create something better than any single agent could produce alone.

## 🤖 Meet the Agent Crew

`better-notes` uses a team of specialized agents that work together:

### 🧠 The Planner
The meta-agent that analyzes your document and creates tailored instructions for all other agents. The Planner considers document type, user preferences, and special requirements to optimize the entire analytical process.

```python
# The Planner creates document-specific instructions
plan = await planner.create_plan(
    document_info=document_info,
    user_preferences=options,
    crew_type="issues"
)
```

### 🔍 The Extractor
Identifies relevant information from each document chunk, adding initial metadata about location and context. It works in parallel across different chunks of your document.

### 🧩 The Aggregator
Combines similar findings from different chunks, eliminates duplicates, and preserves important variations. It's responsible for ensuring comprehensive coverage without redundancy.

### ⚖️ The Evaluator
Determines the importance, severity, and relationships between findings. This critical-thinking agent transforms raw extractions into evaluated insights by adding rationales and impact assessments.

### 📊 The Formatter
Creates structured, navigable reports optimized for human consumption. It organizes content by priority, creates executive summaries, and enhances readability with visual elements.

### 🔎 The Reviewer
Performs quality control on the final output, ensuring the analysis meets quality standards and aligns with user expectations.

## 🛠️ Core Architectural Patterns

`better-notes` implements four foundational patterns that elevate AI applications:

| Pattern | What It Means | How `better-notes` Uses It |
|---------|---------------|--------------------------|
| **Reflection** | AI assessing its own outputs | Reviewer agent scores analysis quality across multiple dimensions |
| **Tool Use** | AI invoking external capabilities | Agents access configs, document metadata, and processing services |
| **Planning** | AI creating step-by-step strategies | Planner agent designs document-specific instructions for other agents |
| **Collaboration** | Specialized AIs working together | The entire pipeline divides complex analysis into specialized expert tasks |

## 📝 The Journey: Building Through Experimentation

`better-notes` evolved through continuous experimentation:

```
Simple Summarization → Multi-Stage Processing → Agent Specialization → Meta-Planning
```

Each iteration revealed new insights about practical AI system design:

* How specialized agents produce better results than monolithic systems
* The importance of proper context management between processing stages
* Finding the right balance between parallelization and sequential processing
* Making AI-generated content render beautifully in web applications

The journey taught me that building AI products requires constant iteration and willingness to reimagine architecture as you learn. Some of the most interesting discoveries came from unexpected challenges:

> *"The most elegant code solution isn't always the best AI architecture. Sometimes you need to embrace the messy reality of large language models and build around their strengths and limitations."*

## 🚀 Getting Started

```bash
# Clone repository
git clone https://github.com/kris-nale314/better-notes.git
cd better-notes

# Set up environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure and run
cp .env.example .env  # Add your OpenAI API key
streamlit run app.py
```

## 🏗️ Project Structure

```
better-notes/
├── app.py                      # Streamlit UI
├── orchestrator.py             # Coordinates agent crews
├── agents/
│   ├── base.py                 # BaseAgent foundation
│   ├── planner.py              # Meta-planning agent
│   ├── extractor.py            # Extracts from chunks
│   ├── aggregator.py           # Combines chunk results
│   ├── evaluator.py            # Assesses importance
│   ├── formatter.py            # Creates reports
│   └── reviewer.py             # Quality control
├── crews/
│   ├── issues_crew.py          # Issues analysis crew
├── lean/
│   ├── chunker.py              # Macro-chunking logic
│   └── async_openai_adapter.py # LLM interface
└── ui_utils/
    ├── core_styling.py         # UI styling
    └── result_formatting.py    # Output enhancement
```

## 💬 Try it yourself!

`better-notes` works best with:
* Meeting transcripts
* Technical documents
* Business reports
* Research papers
* Project documentation

Upload your documents and experiment with different detail levels and focus areas. Try the chat interface to explore your document after analysis. Each interaction helps you discover new insights from your content.

## 🔮 What's Next?

`better-notes` is both a functional tool and an experimental playground. Future directions include:

* Action Items crew for task extraction
* Meeting Insights crew for participant and decision analysis 
* Fine-tuned extraction models for specialized domains
* User feedback loops to improve agent performance
* Expanded visualization options for different analysis types

## 📚 Learn More

Interested in how `better-notes` works under the hood? Check out:
* [Technical Architecture](docs/betterNotesArch.md) - Deep dive into the agent system
* [Configuration Guide](docs/betterNotesConfig.md) - How to customize agent behavior


## 🤝 Contribution

Contributions welcome! Whether you're interested in AI architecture, UI improvements, or new analysis types, `better-notes` provides a foundation for experimentation.

<div align="center">

---

<p>better-notes is released under the MIT License</p>
<p>Built with 💙 and lots of experimentation</p>

</div>