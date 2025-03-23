<div align="center">
  <img src="https://raw.githubusercontent.com/kris-nale314/better-notes/main/docs/images/logo.svg" alt="Better-Notes logo" width="180px"/>
  <h1>Better-Notes</h1>
  <h3>Agentic Document Analysis for Meaningful Insights</h3>
  <p>Transform documents and meeting transcripts into structured insights using<br/>a team of specialized AI agents that collaborate, reason, and adapt.</p>
</div>

<p align="center">
  <a href="#-what-makes-this-different">Key Features</a> •
  <a href="#-the-architecture">Architecture</a> •
  <a href="#-the-agent-crew-system">Agent Crews</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-project-structure">Structure</a> •
  <a href="#-why-its-interesting">Benefits</a>
</p>

---

## 🧠 What Makes This Different

**Better Notes** isn't just another summarizer. It's a modular, multi-agent system that works together like a team of specialists to analyze documents and deliver insights you can actually use:

<table>
<tr>
  <td width="50%" valign="top">
    <h4>🤖 Specialized Agents</h4>
    Each agent is optimized for a specific task (extraction, aggregation, evaluation, formatting, review)
    <h4>📊 Metadata Layering</h4>
    Each agent adds and refines metadata, creating progressively richer analysis
    <h4>🎛️ JSON-Configurable Crews</h4>
    Different analysis teams defined through configuration rather than code changes
  </td>
  <td width="50%" valign="top">
    <h4>📝 Macro-Chunking</h4>
    Handles documents up to 100k tokens by processing in 10k-token macro-chunks (very different from RAG)
    <h4>🧩 Planning Capability</h4>
    The Planner agent acts as a "meta-agent" that optimizes the analysis approach for each document
    <h4>💬 Post-Analysis Chat</h4>
    Interact with your documents through natural conversation using the analysis context
  </td>
</tr>
</table>

<div align="center">
  <img src="https://raw.githubusercontent.com/kris-nale314/better-notes/main/docs/images/logic.svg" alt="Better-Notes Logic" width="90%"/>

  <em>The Planner Agent creates a tailored approach for each document, coordinating specialized agents that extract, process, and organize information</em>

</div>

## 📊 Features In Action

<table>
<tr>
  <td width="60%">
    <h4>🚨 Issues Analysis</h4>
    <p>Identify problems, challenges, and risks categorized by severity. Each issue includes detailed context, impact assessment, and relationship mapping to other identified issues.</p>

    <h4>🔄 Reanalysis & Refinement</h4>
    <p>Adjust parameters and focus areas to refine insights when you need a different perspective or level of detail from your document.</p>
  </td>
  <td width="40%">
    <h4>🔍 Focus Customization</h4>
    <ul>
      <li><strong>Technical:</strong> Implementation, architecture, infrastructure</li>
      <li><strong>Process:</strong> Workflows, procedures, methodologies</li>
      <li><strong>Resource:</strong> Staffing, budget, time constraints</li>
      <li><strong>Quality:</strong> Performance, standards, testing</li>
      <li><strong>Risk:</strong> Compliance, security, strategic hazards</li>
    </ul>
  </td>
</tr>
</table>

<div align="center">
  <img src="https://via.placeholder.com/800x300?text=Screenshot+of+Better+Notes+Interface" alt="Better Notes Interface" width="90%"/>
  <em>From document upload to structured insight in minutes</em>
</div>

## 🧩 The Architecture

Better Notes implements four foundational patterns that elevate AI applications:

| Pattern | What It Means | How Better Notes Uses It |
|---------|---------------|--------------------------|
| **Reflection** | AI assessing its own outputs | Reviewer agent scores analysis quality across multiple dimensions |
| **Tool Use** | AI invoking external capabilities | Agents access configs, document metadata, and processing services |
| **Planning** | AI creating step-by-step strategies | Planner agent designs document-specific instructions for other agents |
| **Collaboration** | Specialized AIs working together | The entire pipeline divides complex analysis into specialized expert tasks |

### Agent Pipeline Flow

```
flow 
    A[Document Upload] --> B[Macro-Chunking]
    B --> C[Planner Agent]
    C -- Creates tailored instructions --> D
    
    D[Agent Crew]
    D1[Extractor] --> D2[Aggregator]
    D2 --> D3[Evaluator]
    D3 --> D4[Formatter]
    D4 --> D5[Reviewer]
    
    D --> E[Analysis Report]
    E --> F[Post-Analysis Features]
    
    F[Post-Analysis Features]
    F1[Document Chat] 
    F2[Reanalysis Options]
    F3[Technical Insights]
    
```

## 🔄 The Agent Crew System

Assessment Crews are specialized teams of AI agents that work together through a structured, configurable workflow:

<table>
<tr>
  <td width="30%" valign="top">
    <h4>🧠 Planner Agent</h4>
    <p>Determines the optimal approach given the document and user preferences</p>
    <h4>🔍 Extractor Agent</h4>
    <p>Identifies relevant information from each document chunk</p>
    <h4>🧩 Aggregator Agent</h4>
    <p>Combines findings, eliminates duplicates, and enhances metadata</p>
  </td>
  <td width="30%" valign="top">
    <h4>⚖️ Evaluator Agent</h4>
    <p>Assesses importance, priority, and relationships</p>
    <h4>📊 Formatter Agent</h4>
    <p>Creates structured, navigable reports</p>
    <h4>🔎 Reviewer Agent</h4>
    <p>Performs quality assessment across multiple dimensions</p>
  </td>
  <td width="40%" valign="top">
    <h4>Configurable Through JSON</h4>
    
```json
{
  "agents": {
    "extraction": {
      "role": "Issue Extractor",
      "goal": "Identify all potential issues",
      "instructions": "Analyze the document...",
      "output_format": {...}
    }
  }
}
```
  </td>
</tr>
</table>

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
```

## 🔍 Why It's Interesting

Better Notes demonstrates practical applications of several emerging techniques in AI:

<table>
<tr>
  <td width="33%" align="center">
    <h3>🤖</h3>
    <strong>Agent Specialization</strong>
    <p>Different agents optimized for specific subtasks</p>
  </td>
  <td width="33%" align="center">
    <h3>🧠</h3>
    <strong>Planning Layers</strong>
    <p>Meta-agents that coordinate and optimize other agents</p>
  </td>
  <td width="33%" align="center">
    <h3>📊</h3>
    <strong>Metadata Enhancement</strong>
    <p>Progressive enrichment through the pipeline</p>
  </td>
</tr>
<tr>
  <td width="33%" align="center">
    <h3>👥</h3>
    <strong>Crew-Based Architecture</strong>
    <p>Configurable teams with defined workflows</p>
  </td>
  <td width="33%" align="center">
    <h3>🖥️</h3>
    <strong>UI/UX for Agent Systems</strong>
    <p>Clean visualization of complex processes</p>
  </td>
  <td width="33%" align="center">
    <h3>🔄</h3>
    <strong>Adaptive Processing</strong>
    <p>Document-specific analysis approaches</p>
  </td>
</tr>
</table>

It's an experimental project that helps to understand how modular AI systems can be built to evolve and adapt to more complex tasks.

## 📝 Development Status

- [x] Planner agent for dynamic instruction creation
- [x] Issues identification with macro-chunks
- [x] Quality review and assessment
- [x] Post-analysis chat interface
- [ ] Action items extraction crew
- [ ] Meeting insights analysis crew

<div align="center">

## 🤝 Contribution

Contributions welcome! 

## 📃 License

[MIT License](LICENSE)

</div>
