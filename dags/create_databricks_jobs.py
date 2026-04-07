import os
import logging
from dotenv import load_dotenv
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import Task, SparkPythonTask, TaskDependency, JobCluster

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# NOTE: Change this to match the folder path where you cloned the code in your Databricks Workspace
WORKSPACE_BASE_PATH = "/Workspace/Users/your.email@company.com/mlflow-genops-poc"

def get_shared_cluster_spec(key: str) -> JobCluster:
    return JobCluster(
        job_cluster_key=key,
        new_cluster={
            "spark_version": "14.3.x-cpu-ml-scala2.12", # ML Runtime
            "node_type_id": "Standard_DS3_v2",          # Azure Default Small Node
            "num_workers": 1,
            "spark_env_vars": {
                "UC_CATALOG": os.getenv("UC_CATALOG", "main"),
                "UC_SCHEMA": os.getenv("UC_SCHEMA", "default"),
                "UC_MODEL_NAME": os.getenv("UC_MODEL_NAME", "kie_pipeline_model"),
                "MODEL_URI": os.getenv("MODEL_URI", "")
            }
        }
    )

def create_inference_dag(w: WorkspaceClient):
    """
    DAG 1: Designed to be triggered by Azure Storage events (when new files arrive).
    Will execute KIE Inference and store logs.
    """
    cluster_key = "inference_cluster"
    
    # In the future, parameters can accept dynamic inputs like Azure Storage Paths:
    # parameters=["--mode", "inference", "--file_path", "{{job.trigger.file_path}}"]
    task_infer = Task(
        task_key="Run_Dynamic_Batch_Inference",
        description="Triggered by Azure Storage. Executes inference against specific new patient data.",
        job_cluster_key=cluster_key,
        spark_python_task=SparkPythonTask(
            python_file=f"{WORKSPACE_BASE_PATH}/dags/inference_job.py", 
            source="WORKSPACE"
        )
    )

    try:
        job = w.jobs.create(
            name="DAG_1_Inference_Event_Trigger",
            job_clusters=[get_shared_cluster_spec(cluster_key)],
            tasks=[task_infer]
        )
        logger.info(f"SUCCESS! Created Inference DAG ID: {job.job_id}")
    except Exception as e:
        logger.error(f"Failed to create Inference DAG: {e}")

def create_evaluation_dag(w: WorkspaceClient):
    """
    DAG 2: Designed to be run manually or automatically via CI/CD redeployment.
    Deploys the model PyFunc and evaluates it against standard ground truth.
    """
    cluster_key = "eval_cluster"
    
    # 1. Model Packaging & Registration
    task_deploy = Task(
        task_key="Deploy_PyFunc_Model",
        description="Registers the core AI extraction logic to Unity Catalog.",
        job_cluster_key=cluster_key,
        spark_python_task=SparkPythonTask(
            python_file=f"{WORKSPACE_BASE_PATH}/deploy_model.py", 
            source="WORKSPACE"
        )
    )
    
    # 2. Evaluation
    task_eval = Task(
        task_key="Run_Model_Evaluation",
        description="Automated Accuracy Measurement against Ground Truth Volume dataset.",
        depends_on=[TaskDependency(task_key="Deploy_PyFunc_Model")],
        job_cluster_key=cluster_key,
        spark_python_task=SparkPythonTask(
            python_file=f"{WORKSPACE_BASE_PATH}/dags/evaluation_job.py",
            source="WORKSPACE"
        )
    )

    try:
        job = w.jobs.create(
            name="DAG_2_Evaluation_And_Deploy",
            job_clusters=[get_shared_cluster_spec(cluster_key)],
            tasks=[task_deploy, task_eval]
        )
        logger.info(f"SUCCESS! Created Evaluation DAG ID: {job.job_id}")
    except Exception as e:
        logger.error(f"Failed to create Evaluation DAG: {e}")

if __name__ == "__main__":
    try:
        w = WorkspaceClient()
        logger.info("Successfully connected to Databricks Workspace.")
    except Exception as e:
        logger.error(f"Failed to connect. Check DATABRICKS_HOST in .env: {e}")
        exit(1)

    logger.info("==== Building DAG 1 (Triggered Inference) ====")
    create_inference_dag(w)
    
    logger.info("==== Building DAG 2 (CI/CD Evaluation & Deployment) ====")
    create_evaluation_dag(w)
