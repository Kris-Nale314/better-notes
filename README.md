<p align="center">
  <img src="https://via.placeholder.com/250x150?text=Better+Notes" alt="Better Notes Logo" width="250"/>
</p>

<h1 align="center">Better Notes</h1>
<p align="center"><strong>AI-powered document analysis assistant built with agent-based architecture</strong></p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/status-beta-orange" alt="Development Status"></a>
</p>

## Why Agent-Based Analysis Matters in Document Processing

Traditional document processors treat analysis as a single-pass operation, leading to:

- **Depth limitations** when complex topics need specialized attention
- **Context isolation** when insights are processed independently
- **Quality ceilings** when one model tackles multi-faceted problems

Better Notes addresses these issues through multi-agent teams:

- **Specialized agents** focus on their domain of expertise
- **Collaborative processing** enhances output quality
- **Parallel execution** scales to very large documents
- **Iterative refinement** responds to user feedback

## Key Capabilities

<table>
<tr>
<td width="50%">

### Smart Document Analysis
- Structure-aware document processing
- Meeting transcript specialization
- Parallel chunk processing
- Hierarchical synthesis

</td>
<td width="50%">

### Specialized Agent Teams
- Issues identification crew
- Action items extraction crew
- Opportunities discovery crew
- Insight synthesis team

</td>
</tr>
<tr>
<td colspan="2" align="center">
<img src="https://via.placeholder.com/800x400?text=Better+Notes+Architecture" alt="Better Notes Architecture" width="80%"/>
</td>
</tr>
</table>

## File Structure

```
better-notes/
├── agents/
│   ├── __init__.py
│   ├── base.py                # Base agent setup with common functionality
│   ├── extractor.py           # Generic extraction agent
│   ├── aggregator.py          # Generic aggregation agent
│   ├── evaluator.py           # Generic evaluation agent
│   ├── formatter.py           # Generic formatting agent
│   ├── utils/
│   │   └── __init__.py
│   └── config/    
│       └── issues_config.json # Issues identification config
│       └── action_config.json # Action items config
│       └── opp_config.json    # Opportunities config
├── crews/
│   ├── __init__.py
│   ├── issues_crew.py         # Issues identification crew
│   ├── action_crew.py         # Action items crew
│   └── opp_crew.py            # Opportunities crew
├── lean/
│   ├── __init__.py
│   ├── async_openai_adapter.py
│   ├── chunker.py
│   ├── document.py
│   ├── options.py
│   └── synthesizer.py
├── pages/
│   ├── 01_Summary.py          # Original summary page
│   └── 02_Multi_Agent.py      # Generic multi-agent page
├── ui_utils/
│   ├── __init__.py
│   └── refiner.py
├── app.py                     # Main entry point
├── orchestrator.py            # Multi-crew coordinator
└── README.md
```

## How It Works

Better Notes transforms document analysis through crew-based processing:

1. **Document Chunking**: Intelligently segments documents based on structure
2. **Agent Specialization**: Each agent focuses on a specific aspect of analysis
3. **Parallel Processing**: Multiple agents work simultaneously on different chunks
4. **Insight Aggregation**: Results are combined, deduplicated, and organized
5. **Quality Evaluation**: Outputs are assessed for severity, impact, and relevance
6. **Presentation Formatting**: Findings are structured into clear, actionable formats

## Quick Start

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

- **Meeting Transcript Analysis**: Convert lengthy meeting transcripts into structured notes with issues, actions, and opportunities
- **Research Document Processing**: Extract key findings and potential research directions from academic papers
- **Business Report Evaluation**: Identify critical issues and strategic opportunities from quarterly reports
- **Project Documentation Review**: Extract action items and potential problems from project documentation

## Dependencies

- CrewAI: Multi-agent orchestration framework
- OpenAI API: Large language model access
- Streamlit: Web interface
- AsyncIO: Parallel processing support

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
<strong>Better notes through better agents.</strong><br>
Divide, conquer, and synthesize your documents through collaborative AI.
</p>