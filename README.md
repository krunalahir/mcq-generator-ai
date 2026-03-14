# 📘 AI-Powered MCQ Generator

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.118.0-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-red.svg)](https://streamlit.io/)

**Generate multiple-choice questions (MCQs) from PDF documents using AI.**

This application leverages **Mistral AI**, **LangChain**, **FAISS vector search**, and **semantic similarity** to automatically generate contextually relevant MCQs from your PDF documents. Perfect for educators, trainers, and anyone looking to create assessments quickly.

---

## ✨ Features

### 🎯 Core Capabilities
- **PDF Processing**: Upload and process PDF documents of up to 25MB
- **AI-Powered Question Generation**: Uses Mistral AI API for intelligent question creation
- **RAG (Retrieval-Augmented Generation)**: Smart context extraction using FAISS vector search
- **Semantic Distractor Generation**: Incorrect options using KeyBERT and sentence embedding transformer
- **Quality Validation**: Built-in question quality scoring and validation
- **Intelligent Caching**: Results cached for 1 hour to avoid redundant API calls

### 🔒 Security Features
- File upload validation and sanitization
- PDF header validation for file integrity
- Path traversal prevention
- Code injection protection
- Malicious content detection
- File size limits (25MB max)

### 🎨 User Experience
- Clean, modern Streamlit UI with responsive layout
- Real-time progress indicators
- Detailed error messages and validation feedback
- CSV export for easy integration with LMS systems
- Interactive question display with options

### 📊 Technical Highlights
- **Embedding Model**: `all-MiniLM-L6-v2` for semantic similarity
- **Vector Search**: FAISS with MMR (Maximal Marginal Relevance)
- **Text Splitting**: 800 character chunks with 100 character overlap
- **Logging**: Comprehensive logging to console and file

---

## 🏗️ Project Structure

```
MCQ_generator/
├── api.py                  # FastAPI backend server
├── Frontend.py             # Modern Streamlit frontend
├── langchain_core.py       # Core MCQ generation logic
├── test_models.py          # Unit tests
├── requirements.txt        # Python dependencies
├── runtime.txt            # Python runtime version (3.11.9)
├── .gitignore             # Git ignore rules
├── .env                   # Environment variables (create from .env.example)
├── mcq_generator.log      # Application logs
└── README.md              # This file
```

---

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager
- Mistral AI API key ([Get one here](https://mistral.ai))

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd MCQ_generator
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate on macOS/Linux
source .venv/bin/activate

# Activate on Windows
.venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Create a `.env` file in the project root:
```bash
# Mistral AI Configuration
MISTRAL_API_KEY=your_actual_api_key_here

# Optional: Customize settings
MISTRAL_MODEL=mistral-medium
MISTRAL_TEMPERATURE=0.6
MISTRAL_MAX_TOKENS=128

# Backend Configuration
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000

# File Upload Configuration
MAX_FILE_SIZE=25

# Cache Configuration (seconds)
CACHE_TIMEOUT=3600

# Logging
LOG_LEVEL=INFO
```

---

## 🚀 Usage

### Running the Application

The application consists of two components that need to run simultaneously:

#### 1. Start the Backend Server (Terminal 1)
```bash
uvicorn api:app --reload --port 8000
```

The FastAPI server will start at `http://127.0.0.1:8000`

#### 2. Start the Frontend (Terminal 2)
```bash
streamlit run Frontend.py
```

The Streamlit app will open in your browser at `http://localhost:8501`

### Alternative: Legacy Standalone Mode
For a simpler setup without backend/frontend separation:
```bash
streamlit run app.py
```
⚠️ **Note**: The standalone mode uses local FLAN-T5 model instead of Mistral AI and is deprecated.

---

## 📖 How It Works

### Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Streamlit UI  │────▶│  FastAPI Backend │────▶│  MCQ Generator  │
│  (Frontend.py)  │     │     (api.py)     │     │ (langchain_core)│
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                          ┌─────────────────────────┐
                                          │   External Services     │
                                          │  - Mistral AI API       │
                                          │  - Sentence Transformers│
                                          │  - FAISS Vector Store   │
                                          └─────────────────────────┘
```

### MCQ Generation Pipeline

1. **PDF Upload & Validation**
   - User uploads PDF through Streamlit UI
   - Backend validates file type, size, and content
   - Temporary file created in system temp directory

2. **Document Processing**
   - PDF loaded using `PyPDFLoader`
   - Text split into chunks (800 chars, 100 overlap)
   - Chunks embedded using `all-MiniLM-L6-v2`
   - FAISS vector store created for semantic search

3. **Context Retrieval**
   - For each answer, retrieve top 4 relevant chunks
   - Uses MMR (Maximal Marginal Relevance) for diversity
   - Context combined for question generation

4. **Question Generation**
   - Context + answer sent to Mistral AI API
   - Model generates contextually relevant question
   - Response cleaned and validated

5. **Distractor Generation**          
    - Keyphrases extracted using KeyBERT                                                                                                                         
    - Semantic similarity calculated for each phrase and embedding with sentence transformer                                                                                                             
    - Top 3 phrases with similarity 0.3-0.75 selected                                                                                                            
    - Options shuffled randomly 

6. **Output & Caching**
   - MCQs formatted as JSON and CSV
   - Results cached for 1 hour
   - CSV available for download

---

## 🔧 API Reference

### Backend Endpoints

#### `POST /generate_mcqs`

Generates MCQs from uploaded PDF.

**Request:**
- **Content-Type**: `multipart/form-data`
- **Parameters:**
  - `file`: PDF file (max 25MB)
  - `answers`: Comma-separated answers string

**Response:**
```json
{
  "mcqs": [
    {
      "answer": "Photosynthesis",
      "question": "What process do plants use to convert sunlight into energy?",
      "option_1": "Photosynthesis",
      "option_2": "Respiration",
      "option_3": "Fermentation",
      "option_4": "Oxidation",
      "correct_option": "option_1"
    }
  ],
  "csv": "answer,question,option_1,option_2,option_3,option_4,correct_option\n..."
}
```

**Status Codes:**
- `200`: Success
- `400`: Bad request (invalid file, empty answers)
- `413`: File too large
- `500`: Server error

---

## 🧪 Testing

Run the unit tests:
```bash
pytest test_models.py -v
```

### Test Coverage
- ✅ Question quality validation
- ✅ Caching mechanism
- ✅ Cache expiration
- ✅ MCQ generation (mocked)
- ✅ Cache key generation

---

## 📝 Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MISTRAL_API_KEY` | - | Your Mistral AI API key (required) |
| `MISTRAL_MODEL` | `mistral-medium` | Mistral model to use |
| `MISTRAL_TEMPERATURE` | `0.6` | Model temperature (0.0-1.0) |
| `MISTRAL_MAX_TOKENS` | `128` | Max tokens in response |
| `BACKEND_HOST` | `127.0.0.1` | Backend server host |
| `BACKEND_PORT` | `8000` | Backend server port |
| `MAX_FILE_SIZE` | `25` | Max upload size in MB |
| `CACHE_TIMEOUT` | `3600` | Cache duration in seconds |
| `LOG_LEVEL` | `INFO` | Logging level |

### Model Parameters (in `langchain_core.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 800 | Text chunk size for splitting |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `VECTOR_SEARCH_K` | 4 | Number of chunks to retrieve |
| `VECTOR_SEARCH_FETCH_K` | 10 | Initial fetch count for MMR |
| `VECTOR_SEARCH_LAMBDA` | 0.5 | MMR diversity parameter |
| `SIMILARITY_MIN` | 0.3 | Min similarity for distractors |
| `SIMILARITY_MAX` | 0.75 | Max similarity for distractors |
| `MIN_DISTRACTORS` | 3 | Minimum distractors per question |

---

## 🛠️ Troubleshooting

### Common Issues

#### ❌ "MISTRAL_API_KEY environment variable not set"
**Solution:** Create a `.env` file with your API key or export it:
```bash
export MISTRAL_API_KEY=your_key_here
```

#### ❌ "Could not connect to the backend server"
**Solution:** Ensure the FastAPI server is running:
```bash
uvicorn api:app --reload --port 8000
```

#### ❌ "Only X distractors generated"
**Solution:** This happens when the PDF doesn't have enough semantically similar terms. Try:
- Using a longer or more detailed PDF
- Adjusting `SIMILARITY_MIN` and `SIMILARITY_MAX` values
- Adding more content-rich documents

#### ❌ Rate limit errors from Mistral API
**Solution:** The application implements automatic retry with exponential backoff. If persistent:
- Reduce the number of answers per request
- Upgrade your Mistral API plan
- Increase delay between requests in code

#### ❌ "File is not a valid PDF"
**Solution:** Ensure the file:
- Has a `.pdf` extension
- Starts with `%PDF` header
- Is not corrupted

---

## 📊 Performance Benchmarks

| Metric | Value |
|--------|-------|
| Average MCQ generation time | 30-60 seconds |
| Cache hit response time | < 1 second |
| Max PDF size | 25 MB |
| Cache duration | 1 hour |
| Concurrent requests | Limited by API key tier |

---

## 🔐 Security Considerations

### Implemented Security Measures
- ✅ Filename sanitization (path traversal prevention)
- ✅ Content-type validation
- ✅ PDF header verification
- ✅ File size limits
- ✅ Input validation (XSS/injection prevention)
- ✅ Dangerous extension blocking
- ✅ Temporary file cleanup

### Best Practices
- Never commit `.env` file to version control
- Use HTTPS in production
- Implement rate limiting for public deployments
- Regular dependency updates
- Monitor API usage and costs

---

## 📚 Dependencies

### Core Libraries
- **FastAPI** (0.118.0): Modern async web framework
- **Streamlit** (1.50.0): Frontend UI framework
- **LangChain** (0.3.27): LLM application framework
- **FAISS** (1.12.0): Vector similarity search
- **Sentence Transformers** (5.1.1): Text embeddings
- **KeyBERT** (0.9.0): Keyword extraction
- **Requests** (2.32.5): HTTP client
- **Pandas** (2.3.3): Data manipulation

### See `requirements.txt` for complete list

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to functions and classes
- Include type hints where applicable
- Write tests for new features

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- [Mistral AI](https://mistral.ai) for the powerful language models
- [LangChain](https://langchain.com) for the LLM orchestration framework
- [FAISS](https://github.com/facebookresearch/faiss) for efficient vector search
- [Streamlit](https://streamlit.io) for the beautiful UI framework
- [Sentence Transformers](https://www.sbert.net/) for semantic embeddings

---

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review the troubleshooting section

---

## 🗺️ Roadmap

- [ ] Support for multiple file formats (DOCX, TXT)
- [ ] Batch PDF processing
- [ ] Custom question templates
- [ ] Question difficulty levels
- [ ] Export to multiple formats (JSON, XML, QTI)
- [ ] LMS integration (Moodle, Canvas)
- [ ] Multi-language support
- [ ] Question bank management
- [ ] Analytics dashboard

---

**Made with ❤️ using AI ,Langchain and Python**
