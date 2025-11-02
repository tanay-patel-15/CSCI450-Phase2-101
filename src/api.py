from fastapi import FastAPI, UploadFile, File
from metrics import compute_metrics_for_model  # relative import, src is PYTHONPATH
import boto3
import os

s3_client = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
bucket_name = os.environ.get("project-models-group102")
models_table = dynamodb.Table(os.environ.get("models"))


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
    s3_client.put_object(Bucket=bucket_name, Key=file.filename, Body=content)

    # Save metadata to DynamoDB
    models_table.put_item(
        Item={
            "model_id": file.filename,
            "size": len(content),
            "category": "MODEL",  # optional, adjust as needed
        }
    )

    return {"filename": file.filename, "size": len(content)}




