# Medical Document KIE Pipeline (Databricks GenOps)

An enterprise-ready, modular MLOps Proof of Concept (POC) explicitly built to run on **Databricks** utilizing the native **Unity Catalog**. 

This architecture leverages a custom `mlflow.pyfunc.PythonModel` to decouple Model Definition from the **Evaluation Pipeline** and the **Batch Inference Pipeline**, adhering to stringent production Databricks ML standards.

---

## Architecture Stack
* **Language:** Python 3.10+
* **MLOps Platform:** Databricks (Targeting `databricks-uc` registry)
* **Core Framework:** MLflow (`pyfunc`, `log_table`, `log_dict`, custom tracing)
* **Orchestration / LLM Logic:** LangGraph (Mocked, traced with `mlflow.langchain.autolog`)
* **OCR Engine:** Azure Document Intelligence / ADE (Mocked)

## Directory Structure

```text
.
├── deploy_model.py                # Standalone script to register the PyFunc model to Databricks
├── main.py                        # Orchestrator supporting multiple execution modes
├── upload_dataset.py              # Utility to seed mock images & JSON datasets to Unity Catalog Volumes
├── requirements.txt               # Databricks SDK & MLflow Dependencies
├── .env.example                   # Databricks Authentication Configuration
└── src/
    └── kie_pipeline/
        ├── __init__.py            
        ├── evaluation.py          # Granular evaluation logic
        ├── inference.py           # Production batch inference loop
        ├── mock_services.py       # Simulated ADE and LangGraph dependencies
        ├── model.py               # MLflow PyFunc model definition (KIEPipelineModel)
        └── registry.py            # Unity Catalog target configuration
```

## Features Deep Dive

1. **Native Unity Catalog Integration**  
   The system binds to the `databricks-uc` registry endpoints seamlessly and formats your model endpoints as `catalog.schema.model_name`.
2. **Granular Evaluation Pipeline**  
   Runs batch evaluation against ground-truth datasets, tracking exact field-level precision (Patient Name, Visit Date, WBC, RBC, HGB). It generates a wide-format `evaluation_by_field.json` DataFrame securely hosted in your Databricks workspace.
3. **Artifact-First Inference**  
   Inference jobs bypass local disk entirely. Serialized dictionaries are saved straight to the Databricks remote MLflow server.

---

## Installation

Create a virtual environment and install the Unity Catalog compliant dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Authentication Setup
Before running, you must configure your `.env` file to communicate with your Databricks workspace. 
Copy `.env.example` to `.env` and fill in your credentials:
```env
MLFLOW_TRACKING_URI=databricks
DATABRICKS_HOST=https://<your-databricks-workspace-url>
DATABRICKS_TOKEN=<your-personal-access-token>

UC_CATALOG=main
UC_SCHEMA=default
UC_MODEL_NAME=kie_pipeline_model
```

---

## Usage

This project explicitly separates the "One-time Infrastructure Registration" layer from the "Continuous Integration / Inference" layer. 

### Step 1: Provision Unity Catalog Dataset
Before evaluating models, you must seed your backend Databricks Volume with mock images and truth datasets. This step interacts via the Databricks SDK to map `/Volumes/` pathways natively.
```bash
python upload_dataset.py
```

### Step 2: Model Registration (Unity Catalog Setup)
Register the baseline MLflow PyFunc architecture directly into your target Databricks Catalog.
```bash
python deploy_model.py
```
*Note: This automatically provisions your local `.env` file with the deployed `MODEL_URI`. Review `.env.example` to see customizable variables.*

### Step 3: Running Pipeline Jobs
Use `main.py` passing the desired `--mode` argument to trigger specific pipeline tasks pointing to Databricks endpoints. It dynamically downloads the correct Databricks Volume datasets uploaded in Step 1!

**1. Inference Mode**  
Takes a batch of unseen raw images, extracts JSON metadata, and logs outputs securely to Databricks.
```bash
python main.py --mode inference
```

**2. Evaluate Mode**  
Evaluates the active Unity Catalog model against an injected dataset.
```bash
python main.py --mode evaluate
```

**3. Evaluate & Deploy Mode**  
Executes the evaluation gating pipeline first; mimicking logic to gate or push a successful ML model into a Production alias.
```bash
python main.py --mode evaluate-deploy
```
