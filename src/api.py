from fastapi import FastAPI, HTTPException, Depends, Request, status, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from mangum import Mangum
import boto3
import os
import re
import logging
import time
from datetime import datetime
from uuid import uuid4

# Internal imports
from src.metrics import compute_metrics_for_model
from src.auth_deps import require_role
from src.auth import router as auth_router
from src.auth_utils import hash_password

# --- Configuration & Setup ---
START_TIME = time.time()
BUCKET_NAME = os.environ.get("BUCKET_NAME", "project-models-group102")
MODELS_TABLE = os.environ.get("MODELS_TABLE", "models")
AUDIT_TABLE = os.environ.get("AUDIT_TABLE", "audit_logs")
USERS_TABLE = os.environ.get("USERS_TABLE", "users")
# Update the defaults to match the YAML example just in case
DEFAULT_ADMIN_EMAIL = os.environ.get("DEFAULT_ADMIN_EMAIL", "ece30861defaultadminuser") 
DEFAULT_ADMIN_PASSWORD = os.environ.get("DEFAULT_ADMIN_PASSWORD", "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE artifacts;")

logger = logging.getLogger("api_logger")
logger.setLevel(logging.INFO)

# AWS Clients
s3_client = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
models_table = dynamodb.Table(MODELS_TABLE)
audit_table = dynamodb.Table(AUDIT_TABLE)
users_table = dynamodb.Table(USERS_TABLE)

app = FastAPI(title="Trustworthy Model Registry")
app.include_router(auth_router)

# --- Pydantic Models (Matching YAML Spec) ---

class ArtifactData(BaseModel):
    url: str

class ArtifactMetadata(BaseModel):
    name: str
    id: str
    type: str

class ArtifactEnvelope(BaseModel):
    metadata: ArtifactMetadata
    data: ArtifactData

class ArtifactQuery(BaseModel):
    name: str
    types: Optional[List[str]] = None

class ArtifactRegEx(BaseModel):
    regex: str

class SimpleLicenseCheckRequest(BaseModel):
    github_url: str

# --- Helper Functions ---

from src.db_setup import create_tables_if_missing

def initialize_default_admin():
    """Ensures a default admin user is present."""
    create_tables_if_missing() # Ensure tables (especially users) exist
    try:
        response = users_table.get_item(Key={"email": DEFAULT_ADMIN_EMAIL})
        if response.get("Item"):
            return
        
        hashed = hash_password(DEFAULT_ADMIN_PASSWORD)
        users_table.put_item(
            Item={
                "email": DEFAULT_ADMIN_EMAIL,
                "password_hash": hashed,
                "role": "admin",
            }
        )
        logger.info(f"Initialized default admin: {DEFAULT_ADMIN_EMAIL}")
    except Exception as e:
        logger.error(f"Failed to initialize admin: {e}")

def log_audit_event(event_type: str, user_payload: dict, details: dict):
    """Logs an audit event."""
    try:
        item = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_payload.get("sub", "unknown"),
            "role": user_payload.get("role", "unknown"),
            "details": details
        }
        audit_table.put_item(Item=item)
    except Exception as e:
        logger.error(f"Audit log failed: {e}")

def clear_dynamodb_table(table_obj, pk_name, sk_name=None):
    """Scans and deletes all items in a table."""
    try:
        scan = table_obj.scan()
        with table_obj.batch_writer() as batch:
            for each in scan.get("Items", []):
                key = {pk_name: each[pk_name]}
                if sk_name:
                    key[sk_name] = each[sk_name]
                batch.delete_item(Key=key)
    except Exception as e:
        logger.error(f"Failed to clear table {table_obj.name}: {e}")

# --- Endpoints ---

@app.get("/health")
async def health():
    """Heartbeat check (BASELINE)"""
    return {"status": "Service reachable."}

@app.get("/tracks")
async def get_tracks():
    """Return the list of tracks the student plans to implement."""
    return {
        "plannedTracks": [
            "Access control track"
        ]
    }

@app.delete("/reset")
async def reset_system(user=Depends(require_role("admin"))):
    """Reset the registry to a system default state. (BASELINE)"""
    try:
        # Clear Tables
        create_tables_if_missing() # Ensure they exist so we can clear them without 404
        clear_dynamodb_table(models_table, "model_id")
        clear_dynamodb_table(audit_table, "timestamp", "event_type")
        
        # --- FIX: Clear Users Table to ensure clean auth state ---
        clear_dynamodb_table(users_table, "email")
        
        # Clear S3
        try:
            objects = s3_client.list_objects_v2(Bucket=BUCKET_NAME)
            if "Contents" in objects:
                delete_keys = [{"Key": o["Key"]} for o in objects["Contents"]]
                s3_client.delete_objects(Bucket=BUCKET_NAME, Delete={"Objects": delete_keys})
        except Exception as e:
            logger.error(f"S3 cleanup error: {e}")

        # Re-init Admin
        # Note: initialize_default_admin() will now successfully recreate 
        # the admin because the table was cleared above.
        initialize_default_admin()
        
        return {"message": "Registry is reset."}
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(status_code=500, detail="Reset failed")

@app.post("/artifact/{artifact_type}", status_code=201)
async def create_artifact(
    artifact_type: str, 
    body: ArtifactData, 
    user=Depends(require_role("admin", "uploader"))
):
    """Register a new artifact. (BASELINE)"""
    
    # Simple validation of URL
    if not body.url:
        raise HTTPException(status_code=400, detail="Missing URL")

    # Reuse metrics logic to parse name/ID and calculate scores
    try:
        metrics = compute_metrics_for_model(body.url)
        # We use the model name from metrics or fallback to the URL slug
        model_name = metrics.get("name") or body.url.split("/")[-1]
        
        # FIXED: Use UUID for truly unique IDs
        artifact_id = str(uuid4())

        # Check if exists (UUID should always be unique, but double-check)
        existing = models_table.get_item(Key={"model_id": artifact_id})
        if existing.get("Item"):
            raise HTTPException(status_code=409, detail="Artifact exists already")

        # Store in DB with lineage structure
        item = {
            "model_id": artifact_id,
            "name": model_name,
            "type": artifact_type,
            "url": body.url,
            "metrics": metrics,
            "lineage": {"parents": [], "children": []},  # Initialize lineage
            "license_info": {"license_id": metrics.get("license", "UNKNOWN")},
            "size": 100,  # Default size in bytes for cost calculation
            "uploaded_by": user.get("sub"),
            "upload_timestamp": datetime.utcnow().isoformat()
        }
        models_table.put_item(Item=item)
        
        # Log Audit
        log_audit_event("CREATE", user, {"id": artifact_id, "url": body.url})

        # Return Envelope
        return {
            "metadata": {
                "name": model_name,
                "id": artifact_id,
                "type": artifact_type
            },
            "data": {
                "url": body.url
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/artifacts")
async def list_artifacts(
    query_list: List[ArtifactQuery], 
    user=Depends(require_role("admin", "uploader", "viewer"))
):
    """Get artifacts from the registry. (BASELINE)"""
    if not query_list:
        raise HTTPException(status_code=400, detail="Missing query")
    
    query = query_list[0] # Spec says "an array with a single artifact_query"
    
    try:
        # Scan table (inefficient but works for small scale)
        response = models_table.scan()
        items = response.get("Items", [])
        
        results = []
        for item in items:
            # Filter logic
            if query.name == "*" or query.name == item.get("name"):
                results.append({
                    "name": item.get("name"),
                    "id": item.get("model_id"),
                    "type": item.get("type", "model")
                })
                
        # Pagination placeholder (offset logic not fully implemented for simplicity)
        return results 
    except Exception as e:
        logger.error(f"List artifacts failed: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@app.post("/artifact/byRegEx")
async def list_artifacts_regex(
    body: ArtifactRegEx, 
    user=Depends(require_role("admin", "uploader", "viewer"))
):
    """Get any artifacts fitting the regular expression (BASELINE)."""
    if not body.regex:
        raise HTTPException(status_code=400, detail="Missing regex")
    
    try:
        pattern = re.compile(body.regex)
        response = models_table.scan()
        items = response.get("Items", [])
        
        results = []
        for item in items:
            name = item.get("name", "")
            if pattern.search(name):
                 results.append({
                    "name": name,
                    "id": item.get("model_id"),
                    "type": item.get("type", "model")
                })
        
        # FIXED: Return 404 for empty results per spec
        if not results:
            raise HTTPException(status_code=404, detail="No artifact found under this regex")
        
        return results
    except re.error:
        raise HTTPException(status_code=400, detail="Invalid Regex")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/artifacts/{artifact_type}/{id}")
async def get_artifact(
    artifact_type: str, 
    id: str, 
    user=Depends(require_role("admin", "uploader", "viewer"))
):
    """Interact with the artifact with this id. (BASELINE)"""
    try:
        response = models_table.get_item(Key={"model_id": id})
        item = response.get("Item")
        
        if not item:
            raise HTTPException(status_code=404, detail="Artifact does not exist")
        
        return {
            "metadata": {
                "name": item.get("name"),
                "id": item.get("model_id"),
                "type": item.get("type")
            },
            "data": {
                "url": item.get("url")
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/artifacts/{artifact_type}/{id}")
async def update_artifact(
    artifact_type: str,
    id: str,
    body: ArtifactEnvelope,
    user=Depends(require_role("admin", "uploader"))
):
    """Update this content of the artifact. (BASELINE)"""
    try:
        # Verify artifact exists
        existing = models_table.get_item(Key={"model_id": id})
        if not existing.get("Item"):
            raise HTTPException(status_code=404, detail="Artifact does not exist")
        
        # Verify name and id match
        if body.metadata.id != id:
            raise HTTPException(status_code=400, detail="ID mismatch")
        
        # Recompute metrics for new URL
        try:
            metrics = compute_metrics_for_model(body.data.url)
        except:
            metrics = {}
        
        # Update the artifact
        item = existing["Item"]
        item.update({
            "name": body.metadata.name,
            "type": body.metadata.type,
            "url": body.data.url,
            "metrics": metrics,
            "updated_by": user.get("sub"),
            "update_timestamp": datetime.utcnow().isoformat()
        })
        models_table.put_item(Item=item)
        
        log_audit_event("UPDATE", user, {"id": id, "url": body.data.url})
        
        return {"message": "Artifact is updated."}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Update failed")
        raise HTTPException(status_code=400, detail="Update failed")

@app.delete("/artifacts/{artifact_type}/{id}")
async def delete_artifact(
    artifact_type: str, 
    id: str, 
    user=Depends(require_role("admin"))
):
    """Delete this artifact. (NON-BASELINE)"""
    try:
        # Check existence
        existing = models_table.get_item(Key={"model_id": id})
        if not existing.get("Item"):
            raise HTTPException(status_code=404, detail="Artifact does not exist")
            
        models_table.delete_item(Key={"model_id": id})
        return {"message": "Artifact is deleted."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/artifact/model/{id}/rate")
async def get_rating(id: str, user=Depends(require_role("admin", "uploader", "viewer"))):
    """Get ratings for this model artifact. (BASELINE)"""
    try:
        response = models_table.get_item(Key={"model_id": id})
        item = response.get("Item")
        
        if not item:
            raise HTTPException(status_code=404, detail="Artifact does not exist")
            
        metrics = item.get("metrics", {})
        
        # Map internal metrics dict to ModelRating schema
        # We map missing latencies to 0.0 for now to satisfy schema
        return {
            "name": item.get("name"),
            "category": "model",
            "net_score": float(metrics.get("net_score", 0)),
            "net_score_latency": float(metrics.get("net_score_latency", 0)),
            "ramp_up_time": float(metrics.get("ramp_up_time", 0)),
            "ramp_up_time_latency": float(metrics.get("ramp_up_time_latency", 0)),
            "bus_factor": float(metrics.get("bus_factor", 0)),
            "bus_factor_latency": float(metrics.get("bus_factor_latency", 0)),
            "performance_claims": float(metrics.get("performance_claims", 0)),
            "performance_claims_latency": float(metrics.get("performance_claims_latency", 0)),
            "license": float(metrics.get("license", 0)),
            "license_latency": float(metrics.get("license_latency", 0)),
            "dataset_and_code_score": float(metrics.get("dataset_and_code_score", 0)),
            "dataset_and_code_score_latency": float(metrics.get("dataset_and_code_score_latency", 0)),
            "dataset_quality": float(metrics.get("dataset_quality", 0)),
            "dataset_quality_latency": float(metrics.get("dataset_quality_latency", 0)),
            "code_quality": float(metrics.get("code_quality", 0)),
            "code_quality_latency": float(metrics.get("code_quality_latency", 0)),
            # Phase 2 metrics placeholders
            "reproducibility": 0.5,
            "reproducibility_latency": 0.0,
            "reviewedness": 0.5,
            "reviewedness_latency": 0.0,
            "tree_score": 0.5,
            "tree_score_latency": 0.0,
            "size_score": metrics.get("size_score", {
                "raspberry_pi": 0, "jetson_nano": 0, "desktop_pc": 0, "aws_server": 0
            }),
            "size_score_latency": 0.0
        }
    except Exception as e:
        logger.exception("Rating fetch error")
        raise HTTPException(status_code=500, detail="Error computing metrics")

@app.get("/artifact/model/{id}/lineage")
async def get_lineage(
    id: str, 
    user=Depends(require_role("admin", "uploader", "viewer"))
):
    """Retrieve the lineage graph for this artifact. (BASELINE)"""
    try:
        response = models_table.get_item(Key={"model_id": id})
        item = response.get("Item")
        
        if not item:
            raise HTTPException(status_code=404, detail="Artifact does not exist")
        
        lineage = item.get("lineage", {"parents": [], "children": []})
        
        # Build nodes and edges
        nodes = []
        edges = []
        visited = set()
        
        def add_node(artifact_id, source="registry"):
            if artifact_id in visited:
                return
            visited.add(artifact_id)
            
            try:
                art = models_table.get_item(Key={"model_id": artifact_id}).get("Item")
                if art:
                    nodes.append({
                        "artifact_id": artifact_id,
                        "name": art.get("name"),
                        "source": source
                    })
                    return art
            except:
                pass
            return None
        
        # Add current artifact
        add_node(id, "registry")
        
        # Add parents and their edges
        for parent_id in lineage.get("parents", []):
            add_node(parent_id, "registry")
            edges.append({
                "from_node_artifact_id": parent_id,
                "to_node_artifact_id": id,
                "relationship": "dependency"
            })
        
        # Add children and their edges
        for child_id in lineage.get("children", []):
            add_node(child_id, "registry")
            edges.append({
                "from_node_artifact_id": id,
                "to_node_artifact_id": child_id,
                "relationship": "dependency"
            })
        
        return {
            "nodes": nodes,
            "edges": edges
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Lineage fetch error")
        raise HTTPException(status_code=400, detail="The lineage graph cannot be computed because the artifact metadata is missing or malformed")

@app.post("/artifact/model/{id}/license-check")
async def check_license_compatibility(
    id: str,
    body: SimpleLicenseCheckRequest,
    user=Depends(require_role("admin", "uploader", "viewer"))
):
    """Assess license compatibility for fine-tune and inference usage. (BASELINE)"""
    try:
        # Get artifact
        response = models_table.get_item(Key={"model_id": id})
        item = response.get("Item")
        
        if not item:
            raise HTTPException(status_code=404, detail="The artifact or GitHub project could not be found")
        
        # Get license info from artifact
        license_info = item.get("license_info", {})
        artifact_license = license_info.get("license_id", "UNKNOWN")
        
        # Permissive licenses that are generally compatible
        PERMISSIVE = {
            "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", 
            "ISC", "Unlicense", "CC0-1.0"
        }
        
        # Check if license is permissive
        is_compatible = artifact_license in PERMISSIVE
        
        return is_compatible
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("License check error")
        raise HTTPException(status_code=400, detail="The license check request is malformed or references an unsupported usage context")

@app.get("/artifact/{artifact_type}/{id}/cost")
async def get_cost(
    artifact_type: str, 
    id: str, 
    dependency: bool = False, 
    user=Depends(require_role("admin", "uploader", "viewer"))
):
    """Get the cost of an artifact (BASELINE)"""
    try:
        response = models_table.get_item(Key={"model_id": id})
        item = response.get("Item")
        if not item:
            raise HTTPException(status_code=404, detail="Artifact does not exist")
        
        # Calculate standalone cost
        size = int(item.get("size", 100))
        standalone_cost = round(size / 1024 / 1024, 2)  # MB
        
        if not dependency:
            # Simple case: just return artifact's own cost
            return {
                id: {
                    "total_cost": standalone_cost
                }
            }
        else:
            # Complex case: traverse lineage and sum parent costs
            result = {}
            visited = set()
            
            def add_artifact_cost(artifact_id):
                if artifact_id in visited:
                    return 0.0
                visited.add(artifact_id)
                
                try:
                    art = models_table.get_item(Key={"model_id": artifact_id}).get("Item")
                    if not art:
                        return 0.0
                    
                    art_size = int(art.get("size", 100))
                    art_cost = round(art_size / 1024 / 1024, 2)
                    
                    # Recursively add parent costs
                    lineage = art.get("lineage", {})
                    parent_cost_sum = sum(
                        add_artifact_cost(parent_id) 
                        for parent_id in lineage.get("parents", [])
                    )
                    
                    total = art_cost + parent_cost_sum
                    
                    result[artifact_id] = {
                        "standalone_cost": art_cost,
                        "total_cost": total
                    }
                    
                    return total
                except Exception as e:
                    logger.error(f"Error processing artifact {artifact_id}: {e}")
                    return 0.0
            
            add_artifact_cost(id)
            return result
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cost calculation error: {e}")
        raise HTTPException(status_code=500, detail="The artifact cost calculator encountered an error")

# Startup init removed in favor of lazy-creation in auth.py to handle race conditions

lambda_handler = Mangum(app)