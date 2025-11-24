from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Depends, Request, status
from fastapi.responses import StreamingResponse, FileResponse
from src.metrics import compute_metrics_for_model  # relative import, src is PYTHONPATH
import boto3
import os
from botocore.exceptions import ClientError
import re
import io
from src.auth_deps import require_role
from src.auth import router as auth_router
import requests
import logging
from datetime import datetime
import httpx

SECURITY_HOOK_URL = os.environ.get("SECURITY_HOOK_URL")
MAX_DOWNLOAD_SIZE_BYTES = int(os.environ.get("MAX_DOWNLOAD_SIZE_BYTES", "524288000"))

logger = logging.getLogger("api_logger")

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
async def ingest(url: str, mark_sensitive: bool = False, user=Depends(require_role("admin", "uploader"))):
    metrics = compute_metrics_for_model(url)
    model_id = metrics.get("name")
    if model_id:
        models_table.put_item(
            Item={
                "model_id": model_id,
                "size": metrics.get("size", 0),
                "category": metrics.get("category", "UNKNOWN"),
                "net_score": metrics.get("net_score"),
                "sensitive": bool(mark_sensitive),
                "uploaded_by": user.get("sub") if isinstance(user, dict) else None,
                "upload_timestamp": datetime.utcnow().isoformat()
            }
        )
    return metrics

@app.post("/upload")
async def upload(file: UploadFile = File(...), user=Depends(require_role("admin", "uploader")), sensitive: bool = False):
    content = await file.read()
    # Upload to S3
    s3_client.put_object(Bucket=BUCKET_NAME, Key=file.filename, Body=content)

    # Save metadata to DynamoDB
    models_table.put_item(
        Item={
            "model_id": file.filename,
            "size": len(content),
            "category": "MODEL",  # optional, adjust as needed
            "sensitive": bool(sensitive),
            "uploaded_by": user.get("sub") if isinstance(user, dict) else None,
            "upload_timestamp": datetime.utcnow().isoformat()
        }
    )

    return {"filename": file.filename, "size": len(content), "sensitive": bool(sensitive)}

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
async def run_security_hook(model_id: str, user_id: str):
    """
    Calls external Node security microservice and returns True/False
    """
    url = "http://security-hook-node:3000/security-check"

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json={"model_id": model_id, "user_id": user_id})
        if response.status_code == 200:
            return False
        
        data = response.json()
        return data.get("approved", False)
async def download_model(model_id: str, request: Request, user=Depends(require_role("admin", "uploader", "viewer"))):
    """
    Download a model file from S3. 
    - Validates model_id
    - Checks DynamoDB for sensitivity
    - If sensitive, enforces stricter RBAC and triggers security hook
    - Streams file from S3
    """
    # Basic validation: allow alphanum, dashes, underscores and dots in model_id.
    # prevent path traversal or suspicious characters
    import re
    if not re.match(r"^[A-Za-z0-9_\-\.\/]+$", model_id):
        raise HTTPException(status_code=400, detail="Invalid model id")

    # Fetch metadata from DynamoDB
    try:
        resp = models_table.get_item(Key={"model_id": model_id})
        item = resp.get("Item")
    except Exception as e:
        logger.exception("DynamoDB lookup failed")
        raise HTTPException(status_code=500, detail="Error reading model metadata")

    if not item:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    is_sensitive = bool(item.get("is_sensitive", False))

    # If sensitive, enforce stricter RBAC: require admin OR uploader with explicit permission.
    # Our require_role already allowed viewer/uploader/admin; use payload to check role
    role = user.get("role") if isinstance(user, dict) else None

    # Trigger security hook metadata dict
    hook_payload = {
        "model_id": model_id,
        "sensitive": is_sensitive,
        "requester": user.get("sub") if isinstance(user, dict) else None,
        "requester_role": role,
        "timestamp": datetime.utcnow().isoformat(),
        "path": str(request.url),
    }

    # If model is sensitive, notify security node and restrict access to admin/uploader
    if is_sensitive:
        # Notify security node (best effort)
        try:
            if SECURITY_HOOK_URL:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(SECURITY_HOOK_URL, json=hook_payload, timeout=5)

            if resp.status_code != 200:
                raise HTTPException(
                    status_code=503,
                    detail="Security validation service unavailable"
                )
        except HTTPException:
            raise
        except Exception:
            logger.exception("Failed to call security hook")
            raise HTTPException(
                status_code=500,
                detail="Security hook error"
            )

        # Strict access control for sensitive models
        if is_sensitive:
            allowed_roles = {"admin", "uploader"}
            if role not in allowed_roles:
                hook_payload.update({
                    "blocked_reason": "insufficient_role",
                    "requester_role": role,
                })

                # notify hook
                try:
                    if SECURITY_HOOK_URL:
                        async with httpx.AsyncClient() as client:
                            await client.post(SECURITY_HOOK_URL, json=hook_payload, timeout=5)
                except Exception:
                    logger.exception("Security hook failed during role block")

                raise HTTPException(
                    status_code=403,
                    detail="Sensitive model — restricted to admin/uploader roles"
                )


    # Verify S3 object exists and check size
    try:
        head = s3_client.head_object(Bucket=BUCKET_NAME, Key=model_id)
        size = head.get("ContentLength", 0)
        if size > MAX_DOWNLOAD_SIZE_BYTES:
            # notify security node of blocked download attempt
            try:
                if SECURITY_HOOK_URL:
                    hook_payload.update({"blocked_reason": "size_limit", "size": size})
                    requests.post(SECURITY_HOOK_URL, json=hook_payload, timeout=5)
            except Exception:
                logger.exception("Failed to call security hook for blocked download")
            raise HTTPException(status_code=413, detail="File too large to download")

    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Model file not found in storage")
    except ClientError as e:
        logger.exception("S3 head_object failed")
        raise HTTPException(status_code=500, detail="Error checking stored model")

    # Stream object directly from S3 to response to avoid local temp file
    try:
        s3_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=model_id)
        body = s3_obj["Body"]

        def iterfile():
            try:
                for chunk in iter(lambda: body.read(4096), b""):
                    yield chunk
            finally:
                body.close()

        headers = {
            "Content-Disposition": f'attachment; filename="{model_id}"'
        }

        return StreamingResponse(iterfile(), media_type="application/octet-stream", headers=headers)
    except Exception:
        logger.exception("Error streaming S3 object")
        raise HTTPException(status_code=500, detail="Failed to retrieve model file from S3")

from mangum import Mangum

lambda_handler = Mangum(app)



