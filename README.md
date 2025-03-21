# Better Notes

## AI-powered document analysis with multi-agent collaboration

Better Notes transforms complex documents into structured insights using specialized AI agents working in collaboration. The system excels at analyzing meeting transcripts, reports, and technical documents to extract valuable information.

## Key Features

- **📝 Document Summarization**: Rich, hierarchical summarization that preserves document structure and meaning
- **⚠️ Issues Identification**: Expert analysis that discovers problems, challenges, and risks
- **✅ Action Item Extraction**: Identifies tasks, commitments, and follow-ups with ownership tracking
- **💬 Interactive Chat**: Engage with your documents through natural conversation

## Project Structure

```
better-notes/
├── app.py                    # Main Streamlit application
├── orchestrator.py           # Unified processing orchestrator
├── agents/                   # Agent implementations
│   ├── __init__.py
│   ├── base.py               # Base agent with shared functionality
│   ├── extractor.py          # Extraction specialist agent
│   ├── aggregator.py         # Aggregation specialist agent
│   ├── evaluator.py          # Evaluation specialist agent
│   ├── formatter.py          # Formatting specialist agent
│   └── config/               # Analysis configurations
│       ├── issues_config.json # Issues analysis configuration
│       └── action_config.json # Action items configuration
├── crews/                    # Agent teams (crews)
│   ├── __init__.py
│   ├── issues_crew.py        # Issues identification crew
│   └── action_crew.py        # Action items crew
├── lean/                     # Core document processing
│   ├── __init__.py
│   ├── async_openai_adapter.py # LLM communication layer
│   ├── booster.py            # Performance-enhanced parallel processing
│   ├── chunker.py            # Document chunking
│   ├── document.py           # Document analysis
│   ├── options.py            # Processing options
│   ├── summarizer.py         # Chunk summarization
│   └── synthesizer.py        # Summary synthesis
├── pages/                    # Streamlit pages
│   ├── 01_Summary.py         # Document summarization
│   └── 02_Issues_Identification.py # Issues identification
├── ui_utils/                 # UI enhancement utilities
│   ├── __init__.py
│   └── ui_enhance.py         # UI styling and visualization
├── outputs/                  # Generated analysis outputs
│   ├── issues/               # Issues analysis reports
│   └── intermediates/        # Intermediate agent outputs
└── .env                      # Environment configuration
```

## Architecture

Better Notes uses a multi-agent approach that outperforms traditional single-model analysis:

1. **Extractor Agents** analyze document chunks in parallel
2. **Aggregator Agent** combines and deduplicates findings
3. **Evaluator Agent** assesses importance using defined criteria
4. **Formatter Agent** transforms results into structured reports

The system leverages the orchestrator to coordinate these agents, manage document processing, and provide a unified interface for different analysis types.

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/better-notes.git
   cd better-notes
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up OpenAI API key**
   ```bash
   cp .env.example .env
   # Edit .env to add your API key
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Usage

1. Navigate to the appropriate analysis page (Issues Identification, Summary, etc.)
2. Upload a text document or paste content
3. Adjust analysis settings as needed
4. Process the document
5. Review the structured results
6. Use the chat interface to explore your document further

## Next Steps

1. **Action Items Page**: Implement dedicated page for action item extraction
2. **Performance Optimizations**: Further improve the Booster module for larger documents
3. **Enhanced Chat Interface**: Add memory and document context awareness to chat
4. **Visual Analytics**: Add charts and visualizations for issue statistics
5. **Custom Agent Config UI**: Create interface for customizing agent prompts
6. **Export Options**: Add additional export formats (PDF, DOCX)
7. **Embeddings Support**: Add vector database integration for improved retrievals

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Better Notes: The power of collaborative AI for document understanding