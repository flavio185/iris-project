# Iris Project

Multiclass classification ML pipeline for the Iris dataset (3 classes, 4 numeric features). Built on a medallion architecture (Bronze/Silver/Gold) with MLflow experiment tracking and FastAPI serving.

## Setup

```bash
# Create Python 3.12 virtual environment
make create_environment

# Install dependencies
make requirements
```

## Code Quality

```bash
make lint      # ruff format --check && ruff check
make format    # ruff check --fix && ruff format
make test      # pytest tests/
```

## Pipeline Commands

### Via Makefile

```bash
make bronze             # Ingest raw CSV from S3 → Parquet
make silver             # Clean data (snake_case, normalize species, float64 types)
make validate           # Great Expectations validation on Silver layer
make feature-pipeline   # Silver → Gold feature engineering
make training-pipeline  # Train LR, RF, SVM; log to MLflow
make full-pipeline      # Run all stages end-to-end
```

### Via CLI (project-cli)

```bash
uv run project-cli bronze
uv run project-cli silver
uv run project-cli validate
uv run project-cli feature-pipeline
uv run project-cli training-pipeline
uv run project-cli inference-pipeline <model_uri> <input_path> <output_path>
uv run project-cli full-pipeline
```

## Data Flow

```
S3 (iris.csv) → Bronze (parquet) → Silver (cleaned) → Gold (features) → Training (MLflow)
```

- **Bronze**: Raw CSV ingestion from `s3://datamasters2025/iris/bronze/iris.csv`
- **Silver**: snake_case columns, strip `Iris-` prefix from species, drop `id`, ensure float64
- **Gold**: Versioned features with metadata (4 numeric features: sepal_length, sepal_width, petal_length, petal_width)
- **Training**: Logistic Regression (lbfgs/multinomial), Random Forest, SVM (rbf). Experiment: `iris-baseline-models`

## Inference

### Batch

```bash
uv run project-cli inference-pipeline "models:/iris-logisticregression/1" input.parquet output.parquet
```

### Online (FastAPI)

```bash
uv run uvicorn iris_project.serving.app:app --host 0.0.0.0 --port 8000
```

POST `/predict`:
```json
{
  "data": [
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
  ]
}
```

Returns species prediction + per-class probabilities (setosa, versicolor, virginica).

## Project Structure

```
iris-project/
├── conf/job_config.json              # Ray job config (schedule, replicas)
├── data_processing/
│   ├── bronze/ingest_bronze.py       # S3 CSV → Parquet
│   ├── silver/clean_data.py          # Cleaning pipeline
│   ├── silver/validate_data.py       # Great Expectations validation
│   └── check_s3.py                   # S3 polling helper
├── iris_project/
│   ├── config.py                     # Paths, S3 prefixes, feature/target constants
│   ├── features/
│   │   ├── engineering.py            # engineer_features(), get_feature_names()
│   │   └── preprocessing.py          # StandardScaler, config save/load
│   ├── modeling/
│   │   ├── models.py                 # LR, RF, SVM factories
│   │   ├── eval.py                   # Multiclass metrics (accuracy, F1, ROC-AUC OVR)
│   │   ├── trainer.py                # train_model(), train_and_evaluate()
│   │   ├── data_loader.py            # Load Gold features, stratified split
│   │   ├── pipeline_builder.py       # sklearn Pipeline (preprocessor + classifier)
│   │   └── mlflow_logger.py          # MLflowExperimentLogger
│   ├── pipelines/
│   │   ├── feature_pipeline.py       # Silver → Gold with versioned metadata
│   │   ├── training_pipeline.py      # Train all models, log to MLflow
│   │   └── inference_pipeline.py     # Batch predictions with 3-class probabilities
│   └── serving/app.py                # FastAPI endpoint
├── tests/                            # ~33 tests (features, modeling, pipelines)
├── .github/workflows/                # CI/CD, model training, data pipeline, predict
├── Makefile
├── make.py                           # Typer CLI (project-cli)
└── pyproject.toml                    # Python 3.12, uv, flit
```

## CI/CD

GitHub Actions workflows:
- **ci.yml** — Lint + test on push; on main: package + upload to S3 + deploy to AKS
- **cd.yml** — Deploy to AKS via Ray job manifests
- **model_training.yml** — Manual dispatch: run training pipeline
- **data_pipeline.yml** — Manual dispatch: bronze → silver → validate
- **predict.yml** — Manual dispatch: batch inference with model_uri, input_path, output_path
