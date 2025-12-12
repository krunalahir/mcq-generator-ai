# MCQ Generator

An AI-powered application that generates multiple-choice questions (MCQs) from PDF documents using Mistral AI API.

## Features

- **PDF Processing**: Upload PDF documents and generate MCQs based on the content
- **Mistral AI Integration**: Uses Mistral API for high-quality question generation
- **Context Retrieval**: Smart context extraction to generate relevant questions
- **Distractor Generation**: AI-generated incorrect options using semantic similarity
- **Quality Validation**: Questions are validated for quality and relevance
- **Caching**: Results are cached to avoid redundant API calls
- **Security**: File upload validation and sanitization
- **Logging**: Comprehensive logging for monitoring and debugging

## Improvements Made

### 1. Architecture & Performance
- Fixed import issues in backend
- Improved model efficiency by switching from local FLAN-T5 to Mistral API
- Added caching mechanisms to avoid redundant computations
- Implemented lazy loading for models to improve startup time

### 2. Security
- Added file upload validation and sanitization
- Implemented checks for malicious file types and content
- Added input validation to prevent code injection
- PDF header validation to ensure file integrity

### 3. User Experience
- Enhanced Streamlit UI with better layout and instructions
- Added detailed setup instructions for API keys
- Implemented quality feedback and validation for generated questions
- Added progress indicators and error handling

### 4. Reliability
- Added comprehensive error handling and validation
- Implemented question quality validation and regeneration
- Added proper logging and monitoring
- Added retry mechanisms for low-quality questions

### 5. Code Quality
- Added unit tests for core functionality
- Improved code organization and documentation
- Added proper exception handling throughout

## Setup

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Get Mistral API key from [mistral.ai](https://mistral.ai)

3. Create a `.env` file in the project root with:
   ```
   MISTRAL_API_KEY=your_actual_api_key_here
   ```

4. Run the backend:
   ```bash
   uvicorn api_server:app --reload --port 8000
   ```

5. Run the frontend:
   ```bash
   streamlit run Frontend.py
   ```

## Usage

1. Upload a PDF document
2. Enter answers (comma-separated) you want questions about
3. Click "Generate MCQs"
4. Review and download the results

## Testing

Run the unit tests with:
```bash
pytest test_models.py -v
```

## Logging

The application logs to both console and `mcq_generator.log` file. The logs include:
- Request details and processing times
- Error information
- Security validation results
- Cache hit/miss information