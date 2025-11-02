from fastapi import FastAPI, UploadFile, File
from src.metrics import compute_metrics_for_model  # relative import, src is PYTHONPATH
import boto3
import os

s3_client = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
BUCKET_NAME = os.environ.get("BUCKET_NAME", "project-models-group102")
MODELS_TABLE = os.environ.get("MODELS_TABLE", "models")

s3_client = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
models_table = dynamodb.Table(MODELS_TABLE)



app = FastAPI(title="Trustworthy Model Registry")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/ingest")
async def ingest(url: str):
    metrics = compute_metrics_for_model(url)
    return metrics

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    # Upload to S3
    s3_client.put_object(Bucket=BUCKET_NAME, Key=file.filename, Body=content)

    # Save metadata to DynamoDB
    models_table.put_item(
        Item={
            "model_id": file.filename,
            "size": len(content),
            "category": "MODEL",  # optional, adjust as needed
        }
    )

    return {"filename": file.filename, "size": len(content)}

@app.get("/models/{model_id}")
async def get_model(model_id: str):
    """Fetch a model's metadata from DynamoDB by ID."""
    try:
        response = models_table.get_item(Key={"model_id": model_id})
        item = response.get("Item")
        if item:
            return item
        else:
            return {"error": f"Model '{model_id}' not found"}
    except Exception as e:
        return {"error": str(e)}




