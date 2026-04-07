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
    catalog = os.getenv("UC_CATALOG", "wks_aisd")
    schema = os.getenv("UC_SCHEMA", "kie_project")
    volume_name = "kie_dataset"
    
    # Native Unity Catalog Volume path format
    volume_path = f"/Volumes/{catalog}/{schema}/{volume_name}"
    
    logger.info(f"Targeting Unity Catalog Volume: {volume_path}")
    
    # 2. Automatically Create Catalog (If missing and user has permissions)
    try:
        w.catalogs.get(catalog)
    except Exception:
        logger.info(f"Catalog '{catalog}' not found. Attempting to create it...")
        try:
            w.catalogs.create(name=catalog)
            logger.info(f"Created Managed Catalog: {catalog}!")
        except Exception as e:
            logger.error(f"Error creating catalog (You may be missing Admin perms): {e}")
            return

    # 3. Automatically Create Schema (If missing)
    try:
        w.schemas.get(full_name=f"{catalog}.{schema}")
    except Exception:
        logger.info(f"Schema '{schema}' not found. Attempting to create it...")
        try:
            w.schemas.create(name=schema, catalog_name=catalog)
            logger.info(f"Created Managed Schema: {catalog}.{schema}!")
        except Exception as e:
            logger.error(f"Error creating schema: {e}")
            return
            
    # 4. Automatically Create the Managed Volume
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
    
    # 5. Create Mock Local Images & Upload
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

    # 6. Create Mock Ground Truth JSON Dataset & Upload
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
    logger.info("Go to: Data Space -> Catalog -> wks_aisd -> kie_project -> kie_dataset to view files in DB UI!")

if __name__ == "__main__":
    main()
