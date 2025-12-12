"""
Unit tests for the MCQ Generator
"""
import pytest
import tempfile
import os
from unittest.mock import patch, mock_open
from models import validate_question_quality, generate_mcqs_from_pdf, _get_cache_key, _get_from_cache, _set_in_cache


def test_validate_question_quality_valid():
    """Test that a valid question gets a good score"""
    question = "What is the capital of France?"
    answer = "Paris"
    context = "France is a country in Europe. The capital of France is Paris."
    
    is_valid, score, feedback = validate_question_quality(question, answer, context)
    
    assert is_valid == True
    assert score > 40  # Should meet the quality threshold
    assert len(feedback) == 0 or all("Question should end with a question mark" not in f for f in feedback)


def test_validate_question_quality_short():
    """Test that a very short question gets a low score"""
    question = "Why?"
    answer = "Paris"
    context = "France is a country in Europe. The capital of France is Paris."
    
    is_valid, score, feedback = validate_question_quality(question, answer, context)
    
    assert is_valid == False
    assert score < 40
    assert "Question is too short or empty" in feedback


def test_validate_question_quality_no_question_mark():
    """Test that a question without a question mark gets feedback"""
    question = "What is the capital of France"
    answer = "Paris"
    context = "France is a country in Europe. The capital of France is Paris."
    
    is_valid, score, feedback = validate_question_quality(question, answer, context)
    
    assert "Question should end with a question mark" in feedback


def test_validate_question_quality_contains_answer():
    """Test that a question containing the answer gets feedback"""
    question = "What is Paris?"
    answer = "Paris"
    context = "France is a country in Europe. The capital of France is Paris."
    
    is_valid, score, feedback = validate_question_quality(question, answer, context)
    
    assert "Question should not directly contain the answer" in feedback


def test_cache_get_set():
    """Test the caching mechanism"""
    # Clear any existing cache
    from models import _cache
    _cache.clear()
    
    # Test setting and getting from cache
    pdf_path = "/test/path.pdf"
    answers = "Paris, London"
    
    cache_key = _get_cache_key(pdf_path, answers)
    test_data = {"mcqs": [{"question": "Test?", "answer": "Test"}]}
    
    _set_in_cache(cache_key, test_data)
    cached_result = _get_from_cache(cache_key)
    
    assert cached_result == test_data


def test_cache_expired():
    """Test that expired cache entries are removed"""
    from models import _cache, _is_cache_valid
    from datetime import datetime, timedelta
    
    # Clear any existing cache
    _cache.clear()
    
    # Add an expired entry manually
    cache_key = "test_key"
    expired_time = datetime.now() - timedelta(seconds=1)  # Expired 1 second ago
    _cache[cache_key] = ("test_data", expired_time)
    
    # Try to get the expired entry
    cached_result = _get_from_cache(cache_key)
    
    assert cached_result is None
    assert cache_key not in _cache


@patch('models.PyPDFLoader')
@patch('models.generate_question_with_mistral')
def test_generate_mcqs_from_pdf_basic(mock_generate_question, mock_pdf_loader):
    """Test basic MCQ generation"""
    # Setup mocks
    mock_pdf_loader.return_value.load.return_value = [
        type('obj', (object,), {'page_content': 'The capital of France is Paris.'})()
    ]
    mock_generate_question.return_value = "What is the capital of France?"
    
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as temp_pdf:
        temp_pdf.write("dummy pdf content")
        temp_pdf_path = temp_pdf.name
    
    try:
        # Test the function
        result = generate_mcqs_from_pdf(temp_pdf_path, "Paris")
        
        # Assertions
        assert "mcqs" in result
        assert len(result["mcqs"]) == 1
        assert result["mcqs"][0]["answer"] == "Paris"
        assert result["mcqs"][0]["question"] == "What is the capital of France?"
        
    finally:
        # Clean up temp file
        os.unlink(temp_pdf_path)


def test_get_cache_key():
    """Test that cache key generation is consistent"""
    pdf_path = "/path/to/test.pdf"
    answers = "Paris, London"
    
    key1 = _get_cache_key(pdf_path, answers)
    key2 = _get_cache_key(pdf_path, answers)
    
    assert key1 == key2
    assert len(key1) == 32  # MD5 hash length in hex


if __name__ == "__main__":
    pytest.main([__file__])