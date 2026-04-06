import os
import logging
import mlflow
from dotenv import load_dotenv, set_key

from src.kie_pipeline.registry import log_and_register_model

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("==== Starting Medical Document Core Model Deployment ====")
    
    # Configure MLflow Dev Server URI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri("databricks-uc")
    
    # Enable Langchain autologging for tracing
    try:
        if hasattr(mlflow, "langchain"):
            mlflow.langchain.autolog()
            logger.info("mlflow.langchain.autolog() enabled.")
    except Exception as e:
        logger.warning(f"Could not enable langchain autologging: {e}")

    # --- Register Core Model ---
    model_uri = log_and_register_model()
    
    env_path = ".env"
    if not os.path.exists(env_path):
        open(env_path, 'a').close()
    set_key(env_path, "MODEL_URI", model_uri)
        
    logger.info(f"Saved MODEL_URI ({model_uri}) to .env")
    logger.info("==== Deployment Setup Complete ====")
