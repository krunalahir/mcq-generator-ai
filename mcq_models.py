# models.py
import os
import requests
import hashlib
import re
import logging
from sentence_transformers import SentenceTransformer, util
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from keybert import KeyBERT
import random
import pandas as pd
import tempfile
import json
from datetime import datetime, timedelta

import dotenv
# Load environment variables from .env file
dotenv.load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)


# ===== Caching mechanisms =====
# Simple in-memory cache to store recently generated MCQs
_cache = {}
# Cache timeout in seconds (set to 1 hour)
CACHE_TIMEOUT = 3600

def _get_cache_key(pdf_path, answers):
    """Generate a unique cache key based on PDF path and answers"""
    content = f"{pdf_path}_{answers}"
    return hashlib.md5(content.encode()).hexdigest()

def _is_cache_valid(timestamp):
    """Check if cache entry is still valid"""
    return datetime.now() < timestamp

def _get_from_cache(key):
    """Get value from cache if still valid"""
    if key in _cache:
        value, timestamp = _cache[key]
        if _is_cache_valid(timestamp):
            return value
        else:
            # Remove expired entry
            del _cache[key]
    return None

def _set_in_cache(key, value):
    """Set value in cache with timestamp"""
    timestamp = datetime.now() + timedelta(seconds=CACHE_TIMEOUT)
    _cache[key] = (value, timestamp)


# ===== Model Initialization =====
# Initialize models - using lazy loading to improve startup time
_embedder = None
_embedding_model = None
_kw_model = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _embedder

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embedding_model

def get_kw_model():
    global _kw_model
    if _kw_model is None:
        _kw_model = KeyBERT("all-MiniLM-L6-v2")
    return _kw_model


def generate_question_with_mistral(context, answer):
    """
    Generate a question based on context and answer using Mistral API
    """
    # Get Mistral API key from environment variable
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable not set")

    url = "https://api.mistral.ai/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Truncate context to reduce API usage and avoid rate limits
    truncated_context = context[:500] if len(context) > 500 else context

    prompt = f"""
    Generate a clear, specific question based on the provided context for which the answer is '{answer}'.

    Context: {truncated_context}

    Requirements:
    1. Question must be directly answerable from the context
    2. Use precise wording that makes '{answer}' the only correct option
    4. Keep the question focused and avoid ambiguity
    """

    payload = {
        "model": "mistral-medium",  # You can change this to "mistral-small" or "mistral-large" based on your preference
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.6,  # Lower temperature for more consistent, focused questions
        "max_tokens": 128
    }

    import time

    # Implement retry with exponential backoff for rate limits
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers)

            # Check for rate limit status code
            if response.status_code == 429:
                if attempt < max_retries - 1:  # Not the last attempt
                    wait_time = (2 ** attempt) + 1  # Exponential backoff
                    logger.warning(f"Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                else:
                    response.raise_for_status()
            else:
                response.raise_for_status()

            result = response.json()
            question = result['choices'][0]['message']['content'].strip()

            # Add a small delay to prevent rate limiting on subsequent calls
            time.sleep(1)
            break  # Success, exit the retry loop
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 and attempt < max_retries - 1:
                wait_time = (2 ** attempt) + 1
                logger.warning(f"Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                time.sleep(wait_time)
                continue
            else:
                raise e

    # Additional validation to ensure we only return the question
    # Sometimes the model might return more than just the question
    lines = question.split('\n')
    # Take the first substantial line that looks like a question
    for line in lines:
        line = line.strip()
        if line and ('?' in line or line.lower().startswith(('what', 'when', 'where', 'who', 'why', 'how', 'which', 'the'))):
            if len(line) > 10:  # Ensure it's not a very short fragment
                return line

    return question  # Fallback to returning the whole response


def validate_question_quality(question, answer, context):
    """
    Validates and scores the quality of a generated question
    Returns a tuple: (is_valid, score, feedback)
    """
    score = 0
    feedback = []

    # Check if question is not empty
    if not question or len(question.strip()) < 5:
        return False, 0, ["Question is too short or empty"]

    # Check if question ends with a question mark
    if not question.strip().endswith('?'):
        feedback.append("Question should end with a question mark")
    else:
        score += 10

    # Check for question words
    question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'which', 'whose', 'whom']
    if any(qw in question.lower() for qw in question_words):
        score += 10
    else:
        feedback.append("Question could benefit from using question words like what, when, where, etc.")

    # Check for answer relevance in context
    if answer.lower() in context.lower():
        score += 20
    else:
        feedback.append("Answer may not be well-supported by the context")

    # Check for keyword relevance
    if answer.lower() in question.lower():
        feedback.append("Question should not directly contain the answer")
    else:
        score += 15

    # Check for appropriate length
    if 10 <= len(question.split()) <= 30:
        score += 15
    else:
        feedback.append("Question length should be between 10-30 words for optimal clarity")

    # Check if question seems to make sense
    # Basic check: ensure there's no incomplete sentence or weird repetition
    if re.search(r'\.\s*\.', question):
        feedback.append("Question appears to have formatting issues")
    else:
        score += 10

    # Final validation: if score is above threshold, consider it valid
    is_valid = score >= 40
    if not is_valid and not feedback:
        feedback.append("Question did not meet quality standards")

    return is_valid, score, feedback


# ===== Core Function =====
def generate_mcqs_from_pdf(pdf_path: str, answers: str):
    """Generates MCQs using context retrieval and Mistral API for question generation"""

    logger.info(f"Starting MCQ generation for PDF: {pdf_path}, with {len(answers.split(',')) if answers else 0} answers")

    # Generate cache key
    cache_key = _get_cache_key(pdf_path, answers)

    # Check if result is in cache
    cached_result = _get_from_cache(cache_key)
    if cached_result:
        logger.info(f"Cache hit for {pdf_path}, returning cached result")
        return cached_result

    # Validate inputs
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not answers or not answers.strip():
        logger.error("Answers string is empty")
        raise ValueError("Answers string cannot be empty")

    try:
        logger.info(f"Loading PDF document: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        if not documents:
            logger.error("No content found in PDF file")
            raise ValueError("No content found in PDF file")

        logger.info(f"PDF loaded successfully. Found {len(documents)} document chunks")
    except Exception as e:
        logger.error(f"Error reading PDF file {pdf_path}: {str(e)}")
        raise ValueError(f"Error reading PDF file: {str(e)}")

    # Split and create retriever
    logger.info("Creating document chunks and vector store...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, get_embedding_model())
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
    )
    logger.info(f"Created vector store with {len(chunks)} chunks")

    answers_list = [a.strip() for a in answers.split(",") if a.strip()]
    all_mcqs = []

    logger.info(f"Generating questions for {len(answers_list)} answers")
    for i, answer in enumerate(answers_list):
        logger.info(f"Processing answer {i+1}/{len(answers_list)}: {answer}")

        retrieved_docs = retriever.get_relevant_documents(answer)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Generate question using Mistral API
        question = generate_question_with_mistral(context, answer)

        # Skip quality validation to speed up processing
        score = 100  # Default score
        feedback = []  # No feedback

        # Add quality metrics to the MCQ
        full_text = "\n\n".join([doc.page_content for doc in documents])
        keyphrases = get_kw_model().extract_keywords(full_text, top_n=30, stop_words='english')

        # Distractor generation
        answer_embedding = get_embedder().encode(answer, convert_to_tensor=True)
        candidates = []
        for phrase, _ in keyphrases:
            if phrase.lower() == answer.lower():
                continue
            phrase_embedding = get_embedder().encode(phrase, convert_to_tensor=True)
            similarity = util.cos_sim(answer_embedding, phrase_embedding).item()
            if 0.3 <= similarity <= 0.75:
                candidates.append((phrase, similarity))

        candidates = sorted(candidates, key=lambda x: -x[1])
        distractors = [phrase for phrase, _ in candidates[:3]]

        while len(distractors) < 3:
            distractors.append("Placeholder")

        options = [answer] + distractors[:3]
        random.shuffle(options)

        mcq = {
            "answer": answer,
            "question": question,
            "option_1": options[0],
            "option_2": options[1],
            "option_3": options[2],
            "option_4": options[3],
            "correct_option": f"option_{options.index(answer) + 1}"
        }
        all_mcqs.append(mcq)
        logger.info(f"Generated MCQ for answer: {answer}")

    logger.info(f"Completed generating {len(all_mcqs)} MCQs")
    df = pd.DataFrame(all_mcqs)
    csv_data = df.to_csv(index=False)
    result = {"mcqs": all_mcqs, "csv": csv_data}

    # Cache the result
    _set_in_cache(cache_key, result)
    logger.info("Results cached successfully")

    return result