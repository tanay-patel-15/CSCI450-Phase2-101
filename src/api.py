from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Depends
from fastapi.responses import StreamingResponse, FileResponse
from src.metrics import compute_metrics_for_model  # relative import, src is PYTHONPATH
import boto3
import os
from botocore.exceptions import ClientError
import re
import io
from src.auth_deps import require_role
from src.auth import router as auth_router

s3_client = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
BUCKET_NAME = os.environ.get("BUCKET_NAME", "project-models-group102")
MODELS_TABLE = os.environ.get("MODELS_TABLE", "models")

models_table = dynamodb.Table(MODELS_TABLE)

app = FastAPI(title="Trustworthy Model Registry")
app.include_router(auth_router)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/ingest")
async def ingest(url: str):
    metrics = compute_metrics_for_model(url)
    return metrics

@app.post("/upload")
async def upload(file: UploadFile = File(...), user=Depends(require_role("admin", "uploader"))):
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
async def get_model(model_id: str, user=Depends(require_role("admin", "uploader", "viewer"))):
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

@app.get("/models")
def list_models(search: str = Query(None, description="Regex to filter models by name"), user=Depends(require_role("admin", "uploader", "viewer"))):
    """
    List all models stored in DynamoDB (optionally filter by regex on model_id).
    """
    table = dynamodb.Table(MODELS_TABLE)

    # Scan DynamoDB (can be paginated if many items)
    response = table.scan()
    models = response.get("Items", [])

    # Optional regex filter
    if search:
        pattern = re.compile(search)
        models = [m for m in models if pattern.search(m.get("model_id", ""))]

    return {"models": models}

@app.get("/download/{model_id}")
def download_model(model_id: str, user=Depends(require_role("admin", "uploader", "viewer"))):
    """
    Download a model file from S3.
    """
    try:
        # Create a temporary local path for the file
        tmp_file_path = f"/tmp/{model_id}"

        # Download the model from S3
        s3_client.download_file(BUCKET_NAME, model_id, tmp_file_path)

        # Return the file as a streaming response
        return FileResponse(tmp_file_path, filename=model_id)

    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        # Catch other errors (permission, network, etc.)
        raise HTTPException(status_code=500, detail=f"Error downloading model: {str(e)}")

from mangum import Mangum

lambda_handler = Mangum(app)



