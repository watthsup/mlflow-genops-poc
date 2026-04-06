import time
import logging
import mlflow
from typing import List

logger = logging.getLogger(__name__)

def run_batch_inference_pipeline(model_uri: str, unseen_images: List[str]) -> None:
    """
    Runs batch inference on a set of unseen images and persists the extracted JSON 
    directly to MLflow artifacts without hitting the local filesystem.
    """
    mlflow.set_experiment("KIE_Batch_Inference_Dev")
    with mlflow.start_run(run_name="Scheduled_Batch_Inference"):
        logger.info(f"--- Starting Batch Inference using Model: {model_uri} ---")
        mlflow.log_param("num_images_processed", len(unseen_images))
        
        # Load Model
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        
        # Performance Tracking
        start_time = time.time()
        predictions = loaded_model.predict(unseen_images)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / len(unseen_images) if unseen_images else 0
        mlflow.log_metric("average_processing_time_sec", avg_time)
        mlflow.log_metric("total_processing_time_sec", end_time - start_time)
        logger.info(f"Avg time per doc: {avg_time:.2f}s")
        
        # Log purely to remote MLflow directly using log_dict
        mlflow.log_dict({"results": predictions}, "inference_outputs/batch_results.json")
        logger.info("Batched JSON payload successfully synced directly into MLflow Artifacts")
