from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Depends, Request, status
from fastapi.responses import StreamingResponse, FileResponse
from botocore.exceptions import ClientError
from src.metrics import compute_metrics_for_model  # relative import, src is PYTHONPATH
from src.auth_deps import require_role
from src.auth import router as auth_router
from datetime import datetime
from time import time
import boto3
import os
import re
import io
import requests
import logging
import httpx

START_TIME = time()
SECURITY_HOOK_URL = os.environ.get("SECURITY_HOOK_URL", "http://localhost/security-hook")
MAX_DOWNLOAD_SIZE_BYTES = int(os.environ.get("MAX_DOWNLOAD_SIZE_BYTES", "524288000"))

logger = logging.getLogger("api_logger")

s3_client = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
BUCKET_NAME = os.environ.get("BUCKET_NAME", "project-models-group102")
MODELS_TABLE = os.environ.get("MODELS_TABLE", "models")
AUDIT_TABLE = os.environ.get("AUDIT_TABLE", "audit_log")

models_table = dynamodb.Table(MODELS_TABLE)
audit_table = dynamodb.Table(AUDIT_TABLE)

app = FastAPI(title="Trustworthy Model Registry")
app.include_router(auth_router)

def log_audit_event(event_type: str, user: dict, details: dict):
    """Logs an audit event to the audit_table in DynamoDB"""
    try:
        iten = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user.get("sub"),
            "uder_role": user.get("role"),
            "details": details
        }
        audit_table.put_item(Item=iten)
    except Exception as e:
        logger.error(f"Failed to write audit log for {event_type}: {e}")

@app.get("/auditlog")
async def get_audit_log(limit: int = 100, user=Depends(require_role("admin"))):
    """Retrieves the most recent audit log entries (Admin only)"""
    try:
        response = audit_table.scan(Limit=limit)
        return {"audit_events": response.get("Items", [])}
    except Exception as e:
        logger.exception("Audit log retrieval failed")
        raise HTTPException(status_code=500, detail="Failed to retrieve audit log")

@app.get("/health")
async def health():
    end_time = time()
    uptime_seconds = int(end_time - START_TIME)

    health_check_latency_ms = int((end_time - START_TIME) * 1000)

    return {
        "status": "ok",
        "uptime_seconds": uptime_seconds,
        "latency_ms": health_check_latency_ms,
        "version": os.environ.get("APP_VERSION", "N/A")
    }

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

def call_security_hook(payload: dict, async_mode=True):

    if async_mode:
        async def async_call():
            async with httpx.AsyncClient() as client:
                return await client.post(SECURITY_HOOK_URL, json=payload, timeout=5)
        return async_call()
    else:
        return requests.post(SECURITY_HOOK_URL, json=payload, timeout=5)

@app.get("/security-hook")
async def run_security_hook(model_id: str, user_id: str, request: Request = None):
    """
    Calls external Node security microservice and returns True/False
    """
    payload = {"model_id": model_id, "user_id": user_id}

    async_mode = True
    if request and request.scope.get("test_client", False):
        async_mode = False

    try:
        if async_mode:
            resp = await call_security_hook(payload, async_mode=True)
        else:
            resp = call_security_hook(payload, async_mode=False)
    except Exception:
        raise HTTPException(status_code=500, detail="Security hook error")

    if resp.status_code != 200:
            return False
    return resp.json().get("approved", False)
    
@app.get("/download/{model_id}")
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
    if not re.match(r"^[A-Za-z0-9_\-\.\/]+$", model_id):
        raise HTTPException(status_code=400, detail="Invalid model id")

    # Fetch metadata from DynamoDB
    try:
        resp = models_table.get_item(Key={"model_id": model_id})
        item = resp.get("Item")
    except Exception:
        logger.exception("DynamoDB lookup failed")
        raise HTTPException(status_code=500, detail="Error reading model metadata")

    if not item:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    is_sensitive = bool(
        item.get("is_sensitive")
        or item.get("sensitive")
        or False)

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

    async_mode = True
    if request.scope.get("test_client", False):
            async_mode = False
    # If model is sensitive, notify security node and restrict access to admin/uploader
    if is_sensitive and SECURITY_HOOK_URL:
        try:
            if async_mode:
                resp = await call_security_hook(hook_payload, async_mode=True)
            else:
                resp = call_security_hook(hook_payload, async_mode=False)

            if resp.status_code != 200:
                raise HTTPException(status_code=503, detail="Security validation service unavailable")
        except HTTPException:
            raise
        except Exception:
            logger.exception("Failed to call security hook")
            raise HTTPException(status_code=500, detail="Security hook error")
        
        allowed_roles = {"admin", "uploader"}
        if role not in allowed_roles:
            hook_payload.update({"blocked_reason": "insufficient_role", "requester_role": role})
            try:
                if SECURITY_HOOK_URL:
                    await call_security_hook(hook_payload, async_mode=True)
                else:
                    call_security_hook(hook_payload, async_mode=False)
            except Exception:
                logger.exception("Security hook failed during role block")
            log_audit_event(
                event_type="DOWNLOAD_FAILED",
                user=user,
                details={"model_id": model_id, "sensitive": is_sensitive}
            )
            raise HTTPException(status_code=403, detail="sensitive model - restricted to admin/uploader roles")


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
            log_audit_event(
                event_type="DOWNLOAD_FAILED",
                user=user,
                details={"model_id": model_id, "size": size, "sensitive": is_sensitive}
            )
            raise HTTPException(status_code=413, detail="File too large to download")

    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Model file not found in storage")
    except ClientError as e:
        logger.exception("S3 head_object failed")
        raise HTTPException(status_code=500, detail="Error checking stored model")

    # Stream object directly from S3 to response to avoid local temp file
    try:
        log_audit_event(
            event_type="DOWNLOAD_SUCCESS",
            user=user,
            details={"model_id": model_id, "size": size, "sensitive": is_sensitive}
        )
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
