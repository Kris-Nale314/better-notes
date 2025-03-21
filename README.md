<p align="center">
  <img src="https://via.placeholder.com/250x150?text=Better+Notes" alt="Better Notes Logo" width="250"/>
</p>

<h1 align="center">Better Notes</h1>
<p align="center"><strong>AI-powered document analysis with multi-agent collaboration</strong></p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/status-beta-orange" alt="Development Status"></a>
</p>

## Intelligent Document Analysis

Better Notes transforms complex documents into structured insights using a unique combination of specialized AI agents working in collaboration. Unlike traditional document processing, our multi-agent approach delivers deeper, more nuanced analysis that scales to documents of any size.

### What Makes Better Notes Different?

Most AI document tools pass text through a single model. Better Notes breaks analysis into specialized tasks, each handled by an AI expert. Just as human teams outperform individuals through specialization, our agent teams deliver superior results through focused expertise.

## Key Analysis Types

<table>
<tr>
<td width="33%">

### 📝 Document Summarization

Smart document summarization with rich, hierarchical structuring that preserves the original document's meaning and organization.

</td>
<td width="33%">

### 🔍 Issues Identification

Expert analysis that discovers problems, challenges, and risks, then evaluates their criticality and impact.

</td>
<td width="33%">

### ✅ Action Item Extraction

Identifies tasks, commitments, and follow-ups with ownership assignment and deadline tracking.

</td>
</tr>
</table>

## Multi-Agent Architecture

<p align="center">
  <img src="https://via.placeholder.com/800x300?text=Agent+Collaboration+Architecture" alt="Agent Collaboration Architecture" width="90%"/>
</p>

Better Notes uses a team of specialized AI agents for each analysis type:

1. **Extractor Agents** analyze document chunks in parallel to identify specific elements
2. **Aggregator Agent** combines and deduplicates findings, tracking mention frequency
3. **Evaluator Agent** assesses importance, severity, and relevance using defined criteria
4. **Formatter Agent** transforms raw data into structured, visually rich reports

This collaborative approach delivers higher quality results while handling documents of any size.

## File Structure

```
better-notes/
├── agents/                    # Agent implementations
│   ├── __init__.py
│   ├── base.py                # Base agent with shared functionality
│   ├── extractor.py           # Extraction specialist agent
│   ├── aggregator.py          # Aggregation specialist agent
│   ├── evaluator.py           # Evaluation specialist agent
│   ├── formatter.py           # Formatting specialist agent
│   ├── utils/                 # Agent utilities
│   │   └── __init__.py
│   └── config/                # Analysis configurations
│       ├── issues_config.json # Issues analysis configuration
│       ├── action_config.json # Action items configuration
│       └── opp_config.json    # Opportunities configuration
├── crews/                     # Agent teams (crews)
│   ├── __init__.py
│   ├── issues_crew.py         # Issues identification crew
│   ├── action_crew.py         # Action items crew
│   └── insights_crew.py       # Context insights crew
├── lean/                      # Core document processing
│   ├── __init__.py
│   ├── async_openai_adapter.py# LLM communication layer
│   ├── chunker.py             # Document chunking
│   ├── document.py            # Document analysis
│   ├── options.py             # Processing options
│   ├── summarizer.py          # Chunk summarization
│   └── synthesizer.py         # Summary synthesis
├── ui_utils/                  # UI enhancement utilities
│   ├── __init__.py
│   ├── refiner.py             # Output refinement
│   └── ui_enhance.py          # UI styling and visualization
├── pages/                     # Streamlit pages
│   ├── 01_Summary.py          # Document summarization
│   └── 02_Multi_Agent.py      # Multi-agent analysis
├── outputs/                   # Generated analysis outputs
├── app.py                     # Main Streamlit application
├── orchestrator.py            # Unified processing orchestrator
└── README.md
```

## Under the Hood

### Document Processing

Document processing flows through an intelligent pipeline:

1. **Initial Analysis**: Extracts document type, tone, and key information
2. **Smart Chunking**: Divides documents based on structure and semantic boundaries
3. **Parallel Processing**: Processes chunks simultaneously for efficiency
4. **Hierarchical Synthesis**: Combines insights while preserving document structure

### Agent Collaboration

Each analysis type uses a specialized crew of agents:

1. **Extraction Phase**: Multiple agents analyze document chunks in parallel
2. **Aggregation Phase**: Findings are combined, deduplicated, and organized
3. **Evaluation Phase**: Results are assessed for importance, severity, and relevance
4. **Formatting Phase**: Insights are transformed into structured, visual reports

### Integration Architecture

The `orchestrator.py` provides a unified interface that:

- Creates consistent LLM clients for all components
- Manages document chunking and distribution
- Coordinates agent crews for specialized analysis
- Supports both traditional summarization and multi-agent workflows
- Controls resource usage and API rate limits

## Getting Started

```bash
# Clone the repository
git clone https://github.com/yourusername/better-notes.git
cd better-notes

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Set up OpenAI API key
cp .env.example .env
# Edit .env to add your API key

# Run the application
streamlit run app.py
```

## Example Use Cases

- **Meeting Transcript Analysis**: Convert lengthy meeting transcripts into structured notes
- **Research Document Processing**: Extract key findings and potential research directions
- **Business Report Evaluation**: Identify critical issues and strategic opportunities
- **Project Documentation Review**: Extract action items and potential problems

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
<strong>Transform documents into actionable insights</strong><br>
Better Notes: The power of collaborative AI for document understanding
</p>