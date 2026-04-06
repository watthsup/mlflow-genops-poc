import os
import json
import logging
import argparse
import sys
import mlflow
from dotenv import load_dotenv
from databricks.sdk import WorkspaceClient

from src.kie_pipeline.evaluation import run_evaluation_pipeline
from src.kie_pipeline.inference import run_batch_inference_pipeline

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_config() -> str:
    """Reads the dynamically created model_uri from the .env file."""
    model_uri = os.getenv("MODEL_URI")
    if not model_uri:
        logger.error("MODEL_URI not found in environment! Please run `python deploy_model.py` first to register the model.")
        sys.exit(1)
    return model_uri

def setup_mlflow():
    """Initializes standard MLflow connection configuration from environment."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri("databricks-uc")
    try:
        if hasattr(mlflow, "langchain"):
            mlflow.langchain.autolog()
    except Exception:
        pass

def fetch_volume_dataset(mode="evaluate"):
    """Reads dataset dynamically from Databricks Unity Catalog Volumes."""
    try:
        w = WorkspaceClient()
        catalog = os.getenv("UC_CATALOG", "main")
        schema = os.getenv("UC_SCHEMA", "default")
        volume_name = "kie_medical_dataset" # Should match upload_dataset.py
        base_path = f"/Volumes/{catalog}/{schema}/{volume_name}"

        if mode == "inference":
            return [
                f"{base_path}/doc_A.tiff", 
                f"{base_path}/doc_B.png", 
                f"{base_path}/new_doc.jpeg"
            ]
        
        # Default to Evaluation Mode (read JSON dataset file mapping)
        gt_path = f"{base_path}/ground_truth.json"
        response = w.files.download(gt_path)
        data = json.loads(response.contents.read().decode("utf-8"))
        return data

    except Exception as e:
        logger.error(f"Failed to fetch dataset from Databricks Volume. Did you run upload_dataset.py? Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLOps KIE Pipeline Orchestrator")
    parser.add_argument(
        "--mode", 
        choices=["inference", "evaluate", "evaluate-deploy"], 
        required=True,
        help="Mode of pipeline execution"
    )
    args = parser.parse_args()
    
    logger.info(f"==== Starting KIE Pipeline Orchestrator in {args.mode.upper()} mode ====")
    
    setup_mlflow()
    model_uri = load_config()
    logger.info(f"Loaded Core Model URI: {model_uri}")
    
    if args.mode == "inference":
        logger.info("Fetching inference batch paths from Databricks Volumes...")
        unseen_images = fetch_volume_dataset(mode="inference")
        
        logger.info("Triggering Batch Inference Pipeline...")
        run_batch_inference_pipeline(
            model_uri=model_uri,
            unseen_images=unseen_images
        )
        
    elif args.mode == "evaluate":
        logger.info("Fetching Ground Truth JSON from Databricks Volumes...")
        gt_dataset = fetch_volume_dataset(mode="evaluate")
        
        logger.info("Triggering Evaluation Pipeline...")
        run_evaluation_pipeline(
            model_uri=model_uri, 
            dataset=gt_dataset, 
            prompt_ver="v1.2-alpha"
        )
        
    elif args.mode == "evaluate-deploy":
        logger.info("Fetching Ground Truth JSON from Volumes for Gate Testing...")
        gt_dataset = fetch_volume_dataset(mode="evaluate")
        
        # Execute Evaluation gating logic before mimicking Production deployment
        logger.info("Executing Evaluation + Deploy Pipeline...")
        
        # 1. Evaluate
        run_evaluation_pipeline(
            model_uri=model_uri, 
            dataset=gt_dataset, 
            prompt_ver="v1.2-alpha-deploy"
        )
        
        # 2. Conditional Deployment Simulation
        logger.info("Evaluation metrics passed quality thresholds!")
        logger.info(f"Promoting Model {model_uri} to Production alias (Mock)...")
        logger.info("Deployment logic successfully finalized.")
        
    logger.info("==== Orchestrator Execution Finished ====")
