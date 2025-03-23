<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/kris-nale314/better-notes/main/docs/images/logo.svg" alt="Better-Notes logo" width="30%"/>

# Better-Notes
## Agentic Document Analysis for Meaningful Insights

> Transform documents and meeting transcripts into structured insights using a team of specialized AI agents that collaborate, reason, and adapt.

## 🧠 What Makes This Different

**Better Notes** isn't just another summarizer. It's a modular, multi-agent system that works together like a team of specialists to analyze documents and deliver insights you can actually use:

- **Specialized Agents**: Each agent is optimized for a specific task (extraction, aggregation, evaluation, formatting, review)
- **Macro-Chunking**: Handles documents up to 100k tokens by processing in 10k-token macro-chunks (very different from RAG's tiny chunks)
- **Metadata Layering**: Each agent adds and refines metadata, creating progressively richer analysis
- **Planning Capability**: The Planner agent acts as a "meta-agent" that optimizes the analysis approach for each document
- **JSON-Configurable Crews**: Different analysis teams defined through configuration rather than code changes

```
    A[Document Upload] --> B[Macro-Chunking]

    B --> C[Planner Agent]
    C -- Creates tailored instructions --> D
    
    D[Agent Crew]
    D1[Extractor] --> D2[Aggregator]
    D2 --> D3[Evaluator]
    D3 --> D4[Formatter]
    D4 --> D5[Reviewer]
    
    D --> E[Analysis + Chat]
```

## 📊 Features In Action

- **Issues Analysis**: Identify problems, challenges, and risks categorized by severity
- **Progressive Metadata**: Each agent adds layers of understanding (mentions, confidence, impact)
- **Post-Analysis Chat**: Ask questions about the document with full context from the analysis
- **Reanalysis & Refinement**: Adjust parameters and focus areas to refine insights
- **Focus Customization**: Target analysis on specific areas (Technical, Process, Resource, etc.)

## 🧩 The Architecture

Better Notes implements four foundational patterns that elevate AI applications:

| Pattern | What It Means | How Better Notes Uses It |
|---------|---------------|--------------------------|
| **Reflection** | AI assessing its own outputs | Reviewer agent scores analysis quality across multiple dimensions |
| **Tool Use** | AI invoking external capabilities | Agents access configs, document metadata, and processing services |
| **Planning** | AI creating step-by-step strategies | Planner agent designs document-specific instructions for other agents |
| **Collaboration** | Specialized AIs working together | The entire pipeline divides complex analysis into specialized expert tasks |

<img src="https://raw.githubusercontent.com/kris-nale314/better-notes/main/docs/images/logic.svg" alt="Better-Notes Logic" width="80%"/>

## 🔄 The Agent Crew System

Assessment Crews are specialized teams of AI agents that work together through a structured, configurable workflow:

1. **Planner Agent** determines the optimal approach given the document and user preferences
2. **Extractor Agent** identifies relevant information from each document chunk
3. **Aggregator Agent** combines findings, eliminates duplicates, and enhances metadata
4. **Evaluator Agent** assesses importance, priority, and relationships
5. **Formatter Agent** creates structured, navigable reports
6. **Reviewer Agent** performs quality assessment across multiple dimensions

Each crew is defined through a JSON configuration file that specifies agent roles, instructions, and outputs:

```json
{
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

## 🚀 Getting Started

```bash
git clone https://github.com/kris-nale314/better-notes.git
cd better-notes
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
│   ├── reviewer.py             # Quality control
│   └── config/
│       ├── issues_config.json  # Configuration for issues
│       └── actions_config.json # Configuration for actions
├── crews/
│   ├── issues_crew.py          # Issues analysis crew
├── lean/
│   ├── chunker.py              # Macro-chunking logic
│   ├── async_openai_adapter.py # LLM interface
├── ui_utils/
│   ├── core_styling.py         # UI styling
│   ├── progress_tracking.py    # Progress visualization
│   ├── result_formatting.py    # Output enhancement
│   └── chat_interface.py       # Post-analysis chat
├── pages/
│   ├── 01_Summary.py           # Document summary page
│   ├── 02_Assess_Issues.py     # Issues analysis page
```

## 🔍 Why It's Interesting

Better Notes demonstrates practical applications of several emerging techniques in AI:

1. **Agent Specialization**: Different agents optimized for specific subtasks
2. **Planning Layers**: Meta-agents that coordinate and optimize other agents
3. **Metadata Enhancement**: Progressive enrichment of content through the pipeline
4. **Crew-Based Architecture**: Configurable teams of agents with defined workflows
5. **UI/UX for Agent Systems**: Clean visualization of complex agent processes

It's an experimental project that helps to understand how modular AI systems can be built to evolve and adapt to more complex tasks.

## 📝 Development Status

- [x] Planner agent for dynamic instruction creation
- [x] Issues identification with macro-chunks
- [x] Quality review and assessment
- [x] Post-analysis chat interface
- [ ] Action items extraction crew
- [ ] Meeting insights analysis crew

## 🤝 Contribution

Contributions welcome! 

## 📃 License

[MIT License](LICENSE)
</td>
</tr>
</table>
```
