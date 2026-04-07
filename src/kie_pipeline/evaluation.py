import logging
import pandas as pd
import mlflow
from typing import List, Dict

logger = logging.getLogger(__name__)

def run_evaluation_pipeline(model_uri: str, dataset: List[Dict], prompt_ver: str) -> None:
    """
    Evaluates the PyFunc model against a ground truth dataset, computing field-level metrics.
    """
    mlflow.set_experiment("I042170_KIE_Evaluation_Dev")
    with mlflow.start_run(run_name=f"Eval_Prompt_{prompt_ver}"):
        logger.info(f"--- Starting Evaluation using Model: {model_uri} ---")
        mlflow.log_param("prompt_version", prompt_ver)
        
        # Load Model
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        
        # Construct predictive batch
        dataset_images = [item["image_name"] for item in dataset]
        predictions = loaded_model.predict(dataset_images)
        
        eval_records = []
        fields = ["patient_name", "visit_date", "wbc", "rbc", "hgb"]
        
        metrics = {f"match_{f}": 0 for f in fields}
        metrics["match_all"] = 0
        
        # Granular Field-Level Evaluation
        for gt, pred in zip(dataset, predictions):
            record = {"image_name": gt["image_name"]}
            all_match = True
            
            for field in fields:
                expected_val = gt.get(field, "")
                predicted_val = pred.get(field, "")
                is_match = (expected_val == predicted_val)
                
                record[f"expected_{field}"] = expected_val
                record[f"predicted_{field}"] = predicted_val
                record[f"match_{field}"] = is_match
                
                if is_match:
                    metrics[f"match_{field}"] += 1
                else:
                    all_match = False
                    
            record["match_all_fields"] = all_match
            if all_match:
                metrics["match_all"] += 1
                
            eval_records.append(record)
            
        num_samples = len(dataset)
        
        # 1. Log Detailed Wide-Format Table
        df_eval = pd.DataFrame(eval_records)
        mlflow.log_table(df_eval, "evaluation_by_field.json")
        logger.info("Logged evaluation_by_field.json DataFrame table to MLflow.")
        
        # 2. Compute and Log Detailed Accuracy Metrics
        for k, v_count in metrics.items():
            accuracy = v_count / num_samples if num_samples > 0 else 0.0
            metric_name = f"accuracy_{k.replace('match_', '')}"
            mlflow.log_metric(metric_name, accuracy)
            
        logger.info("Logged field-level evaluate accuracy metrics to MLflow.")
