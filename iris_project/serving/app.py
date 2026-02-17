from datetime import datetime

from fastapi import FastAPI, HTTPException
from loguru import logger
import mlflow
import pandas as pd
from pydantic import BaseModel

from iris_project.config import IRIS_CLASSES

app = FastAPI(title="Iris Online Inference API")


class IrisFeatureRow(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class IrisFeatureBatch(BaseModel):
    data: list[IrisFeatureRow]


MODEL_URI = "models:/iris-logisticregression/1"
logger.info(f"Loading model from MLflow: {MODEL_URI}")
pipeline = mlflow.sklearn.load_model(MODEL_URI)


@app.post("/predict")
def predict(batch: IrisFeatureBatch):
    if not batch.data:
        raise HTTPException(status_code=400, detail="Empty input data")

    X = pd.DataFrame([row.model_dump() for row in batch.data])

    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)

    results = []
    for i in range(len(X)):
        result = X.iloc[i].to_dict()
        result["prediction"] = y_pred[i]
        for j, cls in enumerate(IRIS_CLASSES):
            result[f"proba_{cls}"] = float(y_proba[i, j])
        result["inference_timestamp"] = datetime.utcnow().isoformat()
        results.append(result)

    return results


@app.get("/health")
def health_check():
    return {"status": "ok", "model_uri": MODEL_URI}
