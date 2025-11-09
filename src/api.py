from fastapi import FastAPI, UploadFile, File, APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
from src.metrics import compute_metrics_for_model  # relative import, src is PYTHONPATH
import boto3
import os
from botocore.exceptions import ClientError
import re
import io

s3_client = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
BUCKET_NAME = os.environ.get("BUCKET_NAME", "project-models-group102")
MODELS_TABLE = os.environ.get("MODELS_TABLE", "models")

models_table = dynamodb.Table(MODELS_TABLE)
router = APIRouter()

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

@router.get("/models")
def list_models(search: str = Query(None, description="Regex filter for model name")):
    try:
        response = models_table.scan() # Get all items
        models = response.get("Items", [])
    
        if search:
            pattern = re.compile(search)
            models = [m for m in models if pattern.search(m.get("model_name", ""))]
        return {"models": models}
    except Exception as e:
        return {"error": str(e)}

@router.get("/download/{model_id}")
def download_model(model_id: str):
    # Retrieve metadata from DynamoDB to check if model is non-sensitive
    try:
        response = models_table.get_item(Key={"id": model_id})
        model = response.get("Item")
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        if model.get("sensitive", False):
            raise HTTPException(status_code=403, detail="Model is sensitive")
        
        file_key = model.get("s3_key")
        obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=file_key)
        return StreamingResponse(obj['Body'], media_type="application/zip")
    except ClientError as e:
        raise HTTPException(status_code=500, detail=str(e))

from mangum import Mangum

lambda_handler = Mangum(app)



