import time
import random
import logging
import mlflow

logger = logging.getLogger(__name__)

@mlflow.trace(name="mock_ade_extract")
def mock_ade_extract(image_path: str) -> str:
    """Mocks Azure Document Intelligence identifying raw text from an image."""
    logger.info(f"Extracting text from {image_path} using mock ADE.")
    time.sleep(0.1)  # Simulate network/processing latency
    return f"OCR result for {image_path}: Patient John Doe, Date 2023-10-27. WBC: 5.5, RBC: 4.8, HGB: 14.2"


@mlflow.trace(name="mock_langgraph_logic")
def mock_langgraph_logic(ocr_text: str) -> dict:
    """Mocks the LLM-driven LangGraph logic to extract structured fields."""
    logger.info("Running mock LangGraph extraction logic...")
    time.sleep(0.2)  # Simulate LLM latency
    
    # Introduce slight randomness for realistic evaluation mismatch
    wbc_value = "5.5" if random.random() > 0.1 else "5.2"
    
    return {
        "patient_name": "John Doe",
        "visit_date": "2023-10-27",
        "wbc": wbc_value,
        "rbc": "4.8",
        "hgb": "14.2"
    }
