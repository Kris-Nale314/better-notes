# 📝 Better-Notes
## Multi-Agent Document Analysis

> Transform meeting transcripts and long documents into structured insights using AI agents that collaborate, reason, and adapt.

## ✨ Overview

**Better Notes** is a modular, multi-agent system for analyzing documents up to 100k tokens to extract insights, identify issues, and organize information. Unlike basic summarizers, it uses specialized AI agents orchestrated by a Planner to create deeper, more structured analysis.

Think of it as a team of AI analysts working together on your document:
- The **Planner** designs a document-specific analysis strategy
- Five **Specialized Agents** execute different parts of the analysis
- The processing runs on **macro-chunks** (10k tokens each) for scale

```
    A[Document Upload] --> B[Macro-Chunking (10k tokens)]
    B --> C[Planner Agent]
    C --> D1[Extractor Agent]
    D1 --> D2[Aggregator Agent]
    D2 --> D3[Evaluator Agent]
    D3 --> D4[Formatter Agent]
    D4 --> E[🧪 Reviewer Agent (Optional)]
    E --> F[Final Report + Metadata + Chat]

```

## 🧠 Key Concepts

### Four Essential Patterns of Agentic AI

Better Notes implements four foundational patterns that level up AI applications:

| Pattern | What It Means | How Better Notes Uses It |
|---------|---------------|--------------------------|
| **Reflection** | AI that assesses its own outputs | Reviewer agent scores analysis quality across multiple dimensions |
| **Tool Use** | AI invoking external capabilities | Agents access configs, document metadata, and processing services |
| **Planning** | AI creating step-by-step strategies | Planner agent designs document-specific instructions for other agents |
| **Collaboration** | Specialized AIs working together | The entire pipeline divides complex analysis into specialized expert tasks |

### 📚 Macro-Chunking for Long Documents

- Processes 50k-100k token documents by dividing them into 10k-token macro-chunks
- Parallel extraction with async processing for speed
- Entirely different approach from RAG's tiny retrievable snippets

### 🧩 Modular Multi-Agent Architecture

- `BaseAgent` provides the foundation for all agents
- `PlannerAgent` creates document-specific instructions
- Specialized agents like `ExtractorAgent` focus on single tasks
- Configurable through JSON files, no code changes needed

</td>
</tr>
<tr>
<td colspan="2" align="center">
<img src="https://raw.githubusercontent.com/kris-nale314/better-notes/main/docs/images/logo.svg" alt="Better-Notes logo" width="80%"/>
</td>
</tr>
</table>


## 💡 Analysis Types

- **Issues Identification**: Find problems, challenges, and risks
- **Action Items** (Coming soon): Extract tasks, assignments, and follow-ups
- **Meeting Insights** (Coming soon): Analyze participants, decisions, and sentiment

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
│   ├── planner.py              # Master planning agent
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
├── pages/
│   ├── 01_Summary.py           # Document summary page
│   ├── 02_Assess_Issues.py     # Issues analysis page
```

## 🧠 Why It Matters

Better Notes demonstrates how AI agents can:

1. **Divide complex problems** into more manageable chunks
2. **Use metadata to enhance quality** at each stage
3. **Adapt to document context** using planning
4. **Process longer content** than standard prompting

It's perfect for building:
- Meeting assistants that understand context
- Document analysis systems that find what matters
- Compliance analyzers that identify risks
- Strategic advisors that organize information

## 📝 Development Status

- [x] Planner agent for dynamic instruction creation
- [x] Issues identification with macro-chunks
- [x] Quality review and assessment
- [x] Post-analysis chat interface
- [ ] Action items extraction crew
- [ ] Meeting insights analysis crew

## 🤝 Contribution

Contributions welcome! Check out [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📃 License

[MIT License](LICENSE)