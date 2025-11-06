# RBI Chatbot - Setup and Troubleshooting Guide

## 🎯 Project Status

✅ **COMPLETED:**
- ✅ Project structure created
- ✅ Core chatbot implementation
- ✅ Evaluation framework with LangSmith
- ✅ Streamlit web interface
- ✅ Demo version (works without API quotas)
- ✅ Dependencies installed
- ✅ Environment configured

⚠️ **CURRENT ISSUE:**
- API quota exceeded for Gemini embeddings (free tier limit reached)

## 🚀 How to Run the Project

### Option 1: Demo Application (Primary - No API quotas needed)
```bash
D:/code/projects/try/.venv/Scripts/python.exe src/demo.py
```
This runs the fully functional demo with predefined RBI Q&A pairs.

### Option 2: Streamlit Web Interface (Recommended)
```bash
D:/code/projects/try/.venv/Scripts/streamlit.exe run src/streamlit_app.py
```

### Option 3: Full Version (Requires API quota - Currently disabled)
The full version requires resolving API quota limits. Use demo.py for demonstration.

## 🔧 Fixing the API Quota Issue

### Problem: Gemini API Free Tier Limits
The error indicates you've exceeded the free tier limits for:
- `embed_content_free_tier_requests`
- Daily, per-minute, and per-user quotas

### Solutions:

#### 1. **Wait for Quota Reset (Easiest)**
- Free tier quotas reset every 24 hours
- Try again tomorrow

#### 2. **Upgrade to Paid Plan (Recommended)**
- Go to [Google AI Studio](https://makersuite.google.com/app/billing)
- Enable billing on your Google Cloud Project
- This removes most rate limits

#### 3. **Optimize Embedding Usage**
- Reduce chunk size in `document_processor.py`
- Process smaller portions of the PDF
- Use local embeddings instead

#### 4. **Use Alternative Embeddings**
Let me create a version that uses free local embeddings:

```python
# Alternative: Use sentence-transformers (free, local)
pip install sentence-transformers
```

## 📊 Evaluation Features

The project includes comprehensive evaluation using LangSmith:

### Run Evaluations:
```bash
D:/code/projects/try/.venv/Scripts/python.exe src/evaluate.py
```

### Evaluation Metrics:
- **QA Accuracy**: Measures correctness of answers
- **Helpfulness Score**: Evaluates response quality
- **Source Coverage**: Checks if responses include sources
- **Confidence Levels**: Tracks model confidence
- **Category Performance**: Performance by question type

### Evaluation Dataset:
- 10+ predefined Q&A pairs from RBI FAQs
- Covers: definitions, regulations, requirements, processes
- Categories: definition, regulation, requirements, types, deposits, supervision

## 🎥 Video Creation Guide

For your Loom video submission:

### 1. **Demo the Working Features** (5-7 minutes)
- Show the demo version working
- Ask various RBI questions
- Explain the responses and sources
- Show conversation history

### 2. **Code Walkthrough** (3-5 minutes)
- Explain the architecture:
  - `document_processor.py`: PDF loading and chunking
  - `chatbot.py`: Core RAG implementation
  - `evaluate.py`: LangSmith evaluation framework
  - `streamlit_app.py`: Web interface
- Show the evaluation dataset
- Explain the vector store approach

### 3. **Evaluation Results** (2-3 minutes)
- Show the evaluation metrics
- Explain how you would improve performance
- Discuss the importance of evaluation in LLM applications

### 4. **Technical Challenges** (1-2 minutes)
- Mention the API quota issue
- Explain how you created the demo workaround
- Show production deployment considerations

## 📦 Project Structure

```
rbi-chatbot/
├── src/
│   ├── main.py              # Main application
│   ├── chatbot.py           # Core chatbot logic
│   ├── document_processor.py # PDF processing
│   ├── evaluate.py          # Evaluation framework
│   ├── streamlit_app.py     # Web interface
│   └── demo.py              # Demo version
├── data/                    # Data and vector stores
├── tests/                   # Unit tests
├── requirements.txt         # Dependencies
├── .env.example            # Environment template
├── .env                    # Your API keys
└── README.md               # Documentation
```

## 🔍 Key Technologies Used

- **LangChain**: RAG framework
- **Google Gemini**: LLM for responses
- **FAISS**: Vector database for similarity search
- **LangSmith**: Evaluation and monitoring
- **Streamlit**: Web interface
- **PyPDF**: Document processing

## 📋 Next Steps

1. **For Immediate Demo**: Use `demo.py` - fully functional
2. **For Production**: Resolve API quotas or use local embeddings
3. **For Video**: Record demo.py showing all features
4. **For Submission**: Create zip file (exclude .env with real keys)

## 🛠 Alternative Solutions

If you want to continue with the full version today, I can:

1. **Implement local embeddings** (sentence-transformers)
2. **Create smaller test dataset** (reduce API calls)
3. **Add caching** to reuse embeddings
4. **Implement chunked processing** (process PDF in batches)

Choose which approach you'd prefer!