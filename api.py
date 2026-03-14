import re
import logging
import dotenv
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from mcq_models import generate_mcqs_from_pdf
from datetime import datetime

# Load environment variables from .env file
dotenv.load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("mcq_generator.log"),
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MCQ Generator API",
    description="API for generating Multiple Choice Questions from PDF documents using AI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def is_safe_filename(filename: str) -> bool:
    """
    Check if the filename is safe to use (prevents path traversal and dangerous extensions)
    """
    # Check for path traversal attempts
    if ".." in filename or "/" in filename or "\\" in filename:
        return False

    # Check for dangerous extensions or file types that could be malicious
    dangerous_extensions = [
        '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js', '.jar',
        '.sh', '.pl', '.php', '.py', '.app', '.gadget', '.msp', '.msi', '.msp'
    ]

    lower_filename = filename.lower()
    for ext in dangerous_extensions:
        if lower_filename.endswith(ext):
            return False

    # Check for valid PDF extension
    if not lower_filename.endswith('.pdf'):
        return False

    # Check for potentially dangerous characters in filename
    dangerous_chars = ['<', '>', ':', '"', '|', '?', '*']
    if any(char in filename for char in dangerous_chars):
        return False

    return True


@app.post("/generate_mcqs")
async def generate_mcqs(file: UploadFile, answers: str = Form(...)):
    """FastAPI route for MCQ generation with enhanced security measures"""

    # Log the request
    start_time = datetime.now()
    logger.info(f"Received MCQ generation request for file: {file.filename}, answers length: {len(answers) if answers else 0}")

    # Validate file upload
    if not file:
        logger.warning("Request with no file uploaded")
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Sanitize filename
    filename = file.filename
    if not filename or not is_safe_filename(filename):
        logger.warning(f"Request with invalid or dangerous filename: {filename}")
        raise HTTPException(status_code=400, detail="Invalid or dangerous filename")

    # Validate content type (mimetype)
    content_type = file.content_type
    if content_type not in ["application/pdf", "application/x-pdf", "application/x-bzpdf", "application/x-gzpdf"]:
        # Additional check using file extension as fallback
        if not filename.lower().endswith('.pdf'):
            logger.warning(f"Request with non-PDF file: {filename}, content-type: {content_type}")
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Validate answers input (prevent code injection)
    if not answers or not answers.strip():
        logger.warning("Request with empty answers")
        raise HTTPException(status_code=400, detail="Answers cannot be empty")

    # Additional validation on answers to prevent injection
    if len(answers) > 5000:  # Limit length to prevent abuse
        logger.warning(f"Request with overly long answers input (length: {len(answers)})")
        raise HTTPException(status_code=400, detail="Answers input is too long")

    # Check for potentially malicious content in answers
    forbidden_patterns = [r'<script', r'javascript:', r'vbscript:', r'on\w+\s*=']
    for pattern in forbidden_patterns:
        if re.search(pattern, answers, re.IGNORECASE):
            logger.warning(f"Request with potentially malicious content in answers: {pattern}")
            raise HTTPException(status_code=400, detail="Answers contain forbidden patterns")

    # Validate file size (limit to 25MB)
    file_content = await file.read()
    if len(file_content) > 25 * 1024 * 1024:  # 25MB
        logger.warning(f"Request with oversized file: {len(file_content)} bytes")
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 25MB")

    # Additional check: verify this is actually a PDF file by checking header
    if len(file_content) < 4 or file_content[:4] != b'%PDF':
        logger.warning(f"Request with file that is not a valid PDF: {filename}")
        raise HTTPException(status_code=400, detail="File is not a valid PDF")

    # Create temporary file with sanitized name
    temp_dir = tempfile.gettempdir()
    # Use a safe, unique filename to avoid conflicts
    safe_filename = f"mcq_temp_{os.getpid()}_{hash(filename) % 1000000}.pdf"
    pdf_path = os.path.join(temp_dir, safe_filename)

    try:
        with open(pdf_path, "wb") as f:
            f.write(file_content)

        logger.info(f"Processing file: {filename}, generating MCQs for {len([a.strip() for a in answers.split(',') if a.strip()])} answers")

        # Generate MCQs using the ml_model module which now uses Mistral API
        result = generate_mcqs_from_pdf(pdf_path, answers)

        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Successfully generated MCQs for {filename}, processing time: {processing_time:.2f}s")

        return result

    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Error processing file {filename}: {str(e)}, processing time: {processing_time:.2f}s")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    finally:
        # Clean up temporary file
        if os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
                logger.debug(f"Temporary file removed: {pdf_path}")
            except Exception as e:
                logger.warning(f"Could not remove temporary file {pdf_path}: {str(e)}")