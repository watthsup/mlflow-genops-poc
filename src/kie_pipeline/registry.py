import os
import logging
import mlflow

from .model import KIEPipelineModel

logger = logging.getLogger(__name__)

def log_and_register_model() -> str:
    """
    Starts an MLflow run, logs the PyFunc model, and registers it to Databricks Unity Catalog.
    """
    # Databricks requires experiments to be housed within a Workspace or Repos path. 
    # Modify this if you wish to use a shared folder instead of a personal one.
    mlflow.set_experiment("/Shared/I042170_KIE_Model_Registry_Dev")
    
    uc_catalog = os.getenv("UC_CATALOG", "wks_aisd")
    uc_schema = os.getenv("UC_SCHEMA", "kie_project")
    uc_model_name = os.getenv("UC_MODEL_NAME", "kie_pipeline_model")
    full_model_name = f"{uc_catalog}.{uc_schema}.{uc_model_name}"

    with mlflow.start_run(run_name="Register_KIE_Pipeline"):
        logger.info(f"Logging KIEPipelineModel PyFunc to Unity Catalog as: {full_model_name}")
        
        # Log the custom PyFunc model and explicitly define registered_model_name 
        # to trigger Unity Catalog registration.
        model_info = mlflow.pyfunc.log_model(
            artifact_path="kie_model",
            registered_model_name=full_model_name,
            python_model=KIEPipelineModel()
        )
        model_uri = model_info.model_uri
        
        logger.info(f"Model logged successfully! Model URI: {model_uri}")
        return model_uri
