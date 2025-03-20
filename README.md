# Better Notes

An AI-powered document analysis and note-taking assistant that transforms documents and transcripts into organized, insightful notes with specialized analysis.

## Features

- **Smart Summarization**: Convert long documents into clear, organized notes
- **Specialized Analysis**:
  - Issue Identification: Find problems, challenges, and concerns
  - Opportunity Identification: Discover potential improvements and opportunities
  - Action Item Extraction: Extract tasks, assignments, and follow-ups
- **Customizable Processing**: Control detail level, model selection, and more
- **Simple Interface**: Upload documents and get enhanced notes with minimal setup

## Setup

### Prerequisites

- Python 3.8+ 
- An OpenAI API key

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/better-notes.git
   cd better-notes
   ```

2. Create a virtual environment
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the required packages
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment variables
   ```bash
   cp .env.example .env
   ```
   
   Edit the `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. Verify your setup
   ```bash
   python test_adapter.py
   ```
   This will test your connection to the OpenAI API to ensure everything is working correctly.

## Usage

1. Start the application
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and go to `http://localhost:8501`

3. Upload a document (text file)

4. Configure processing options in the sidebar

5. Click "Process Document" to generate enhanced notes

## Project Structure

- `app.py`: Main entry point and homepage
- `pages/`: Streamlit pages for different features
  - `01_Summary.py`: Document summarization page
- `lean/`: Core processing logic
  - `async_openai_adapter.py`: OpenAI API communication
  - `booster.py`: Performance enhancement for processing
  - `chunker.py`: Document chunking functionality
  - `document.py`: Document analysis
  - `orchestrator.py`: Process orchestration
  - `summarizer.py`: Chunk summarization
  - `synthesizer.py`: Summary synthesis
- `passes/`: Analysis pass definitions
  - `passes.py`: Pass processor logic
  - `configurations/`: JSON configurations for passes
- `ui_utils/`: UI helper utilities
  - `refiner.py`: Summary refinement tools
- `data/`: Directory for sample data
- `outputs/`: Directory for generated outputs

## Development Notes

- Use `.env` for environment variables and API keys
- Run tests before making significant changes
- Keep the architecture modular to easily add new passes

## License

[MIT License](LICENSE)
## File Structure

```
note-summarizer-v2/
├── .env                (with your OPENAI_API_KEY)
├── app.py              (Simple Streamlit entry point)
├── lean/               (Core processing logic)
│   ├── init.py
│   ├── async_openai_adapter.py
│   ├── booster.py
│   ├── chunker.py
│   ├── document.py
│   ├── factory.py
│   ├── itemizer.py     (will become a pass later)
│   ├── options.py
│   ├── refiner.py
│   ├── summarizer.py
│   └── synthesizer.py
├── pages/              (Streamlit pages)
│   ├── 1_Home.py
│   ├── 2_Issue_Analysis.py
│   └── ...             (add other pass pages as needed)
├── passes/             (Pass definitions and configurations)
│   ├── init.py
│   ├── passes.py
│   └── configurations/
│       ├── issue_identification.json
│       ├── opportunity_identification.json
│       └── ...        (add other pass configs as needed)
└── README.md           (Project description and instructions).
```