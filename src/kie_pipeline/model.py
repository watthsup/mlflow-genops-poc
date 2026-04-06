import logging
import mlflow.pyfunc
from typing import List, Dict

from .mock_services import mock_ade_extract, mock_langgraph_logic

logger = logging.getLogger(__name__)

class KIEPipelineModel(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow PyFunc model that encapsulates the complete KIE pipeline.
    It takes an image path, performs ADE extraction, and routes through LangGraph logic.
    """
    def predict(self, context, model_input: List[str]) -> List[Dict[str, str]]:
        """
        Executes KIE Pipeline for a batch of images.
        
        Args:
            context: the MLflow context (injected by MLflow upon load)
            model_input (List[str]): List of image URIs/paths.
            
        Returns:
            List[Dict]: List of dictionaries containing exactly 5 extracted fields.
        """
        results = []
        for image_path in model_input:
            logger.info(f"[Inference Pipeline] Processing: {image_path}")
            ocr_text = mock_ade_extract(image_path)
            extracted_json = mock_langgraph_logic(ocr_text)
            results.append(extracted_json)
        return results
