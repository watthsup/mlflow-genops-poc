import os
import json
import logging
from dotenv import load_dotenv
from databricks.sdk import WorkspaceClient

# Load variables from .env
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    logger.info("==== Starting Databricks Volume Upload Utility ====")
    
    # 1. Initialize Databricks Workspace Client
    # This automatically picks up DATABRICKS_HOST and DATABRICKS_TOKEN from the environment
    try:
        w = WorkspaceClient()
        logger.info("Successfully connected to Databricks Workspace.")
    except Exception as e:
        logger.error(f"Failed to initialize Databricks client. Ensure .env is configured: {e}")
        return

    # Configuration for targeting Unity Catalog Volume
    catalog = os.getenv("UC_CATALOG", "main")
    schema = os.getenv("UC_SCHEMA", "default")
    volume_name = "kie_medical_dataset"
    
    # Native Unity Catalog Volume path format
    volume_path = f"/Volumes/{catalog}/{schema}/{volume_name}"
    
    logger.info(f"Targeting Unity Catalog Volume: {volume_path}")
    
    # 2. Automatically Create the Managed Volume (if it doesn't exist)
    try:
        w.volumes.create(
            catalog_name=catalog,
            schema_name=schema,
            name=volume_name,
            volume_type="MANAGED"
        )
        logger.info(f"Created Managed Volume: {volume_name}!")
    except Exception as e:
        # Expected if the volume already exists or if permissions restrict creation
        logger.info(f"Volume check passed (Already exists or missing creation perms: {e})")
    
    # 3. Create Mock Local Images & Upload
    mock_images = ["doc_A.tiff", "doc_B.png", "new_doc.jpeg"]
    for img in mock_images:
        local_path = f"/tmp/{img}"
        
        # Write dummy binary data locally
        with open(local_path, "wb") as f:
            f.write(b"Mock Medical Image Byte Data")
            
        # Upload via SDK direct to the Volume
        target_path = f"{volume_path}/{img}"
        try:
            with open(local_path, "rb") as f:
                w.files.upload(target_path, f)
            logger.info(f"Successfully uploaded image: {target_path}")
        except Exception as e:
            logger.error(f"Failed to upload {img}: {e}")

    # 4. Create Mock Ground Truth JSON Dataset & Upload
    # This maps the volume pathways exactly how main.py expects them in evaluation mode.
    gt_data = [
        {
            "image_name": f"{volume_path}/doc_A.tiff", 
            "patient_name": "John Doe", 
            "visit_date": "2023-10-27",
            "wbc": "5.5",
            "rbc": "4.8",
            "hgb": "14.2"
        },
        {
            "image_name": f"{volume_path}/doc_B.png", 
            "patient_name": "John Doe", 
            "visit_date": "2023-10-27",
            "wbc": "5.5",
            "rbc": "4.8",
            "hgb": "14.2"
        }
    ]
    
    gt_local_path = "/tmp/ground_truth.json"
    with open(gt_local_path, "w") as f:
        json.dump(gt_data, f, indent=4)
        
    gt_target_path = f"{volume_path}/ground_truth.json"
    try:
        with open(gt_local_path, "rb") as f:
            w.files.upload(gt_target_path, f, overwrite=True)
        logger.info(f"Successfully uploaded Ground Truth Mapping: {gt_target_path}")
    except Exception as e:
        logger.error(f"Failed to upload ground truth: {e}")
    
    logger.info("==== Upload Utility Complete! ====")
    logger.info("Go to: Data Space -> Catalog -> main -> default -> kie_medical_dataset to view files in DB UI!")

if __name__ == "__main__":
    main()
