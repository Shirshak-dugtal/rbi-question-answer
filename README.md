# RBI Chatbot with LangChain and Gemini API

This project implements a chatbot that can answer questions from RBI (Reserve Bank of India) documentation using LangChain, Gemini API, and includes comprehensive evaluations using LangSmith.

## 🚀 Quick Start

### Option 1: Demo Application (Primary)
```bash
python src/demo.py
```

### Option 2: Web Interface (Recommended)
```bash
streamlit run src/streamlit_app.py
```

### Option 3: Run Evaluations
```bash
python src/evaluate.py
```

## Features

- PDF document processing and text chunking
- Vector database using FAISS for semantic search
- Gemini API integration for natural language processing
- LangSmith evaluation framework
- Interactive Streamlit web interface
- Conversation history tracking
- **Demo mode with predefined responses**

## Setup

1. Create and activate virtual environment:
```bash
python -m venv rbi_chatbot_env
# Windows
rbi_chatbot_env\Scripts\activate
# Linux/macOS
source rbi_chatbot_env/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your Gemini API key: `GEMINI_API_KEY=your_api_key_here`
   - Add your LangSmith API key: `LANGCHAIN_API_KEY=your_langsmith_key_here`

## Usage

### Run the main chatbot:
```bash
python src/main.py
```

### Run evaluations:
```bash
python src/evaluate.py
```

### Launch Streamlit interface:
```bash
streamlit run src/streamlit_app.py
```

## Project Structure

- `src/` - Main source code
  - `demo.py` - Main demo application (primary entry point)
  - `chatbot.py` - Core chatbot functionality
  - `document_processor.py` - PDF loading and text processing
  - `evaluate.py` - Evaluation framework
  - `streamlit_app.py` - Web interface
- `data/` - Data files and vector store (auto-created)
- `tests/` - Test files
- `requirements.txt` - Python dependencies
- `.env.example` - Environment variables template

## Data Sources

- RBI Notification PDF: [106MDNBFCS1910202343073E3EF57A4916AA5042911CD8D562.PDF](https://rbidocs.rbi.org.in/rdocs/notification/PDFs/106MDNBFCS1910202343073E3EF57A4916AA5042911CD8D562.PDF)
- FAQ Data: [RBI FAQs](https://www.rbi.org.in/commonman/english/scripts/FAQs.aspx?Id=1167)

## Technologies Used

- **LangChain**: Framework for building LLM applications
- **Google Gemini API**: Large language model for text generation
- **FAISS**: Vector database for similarity search
- **LangSmith**: Evaluation and monitoring
- **Streamlit**: Web interface
- **PyPDF**: PDF processing