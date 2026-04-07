import os
import json
import logging
from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)

def fetch_volume_dataset(mode="evaluate"):
    """Reads dataset dynamically from Databricks Unity Catalog Volumes."""
    try:
        w = WorkspaceClient()
        catalog = os.getenv("UC_CATALOG", "wks_aisd")
        schema = os.getenv("UC_SCHEMA", "kie_project")
        volume_name = "kie_dataset" # Should match upload_dataset.py
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
        logger.error(f"Failed to fetch dataset from Databricks Volume. Error: {e}")
        return []
