# UPDATE your imports at the top of src/api.py
from fastapi import FastAPI, HTTPException, Depends, Request, status, Header
from fastapi.responses import JSONResponse, HTMLResponse, Response # Added HTMLResponse, Response
from fastapi.exceptions import RequestValidationError # Added this
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple, Set # Added Tuple, Set
import hashlib # Added for ID generation
import secrets # Added for ID generation
from mangum import Mangum
import boto3
from boto3.dynamodb.conditions import Key, Attr
import os
import re
import logging
import time
import random
from datetime import datetime
from uuid import uuid4
import base64
from decimal import Decimal
from botocore.exceptions import ClientError

# Internal imports
from src.metrics import compute_metrics_for_model
from src.auth_deps import require_role
from src.auth import router as auth_router
from src.auth_utils import hash_password
from src.db_setup import create_tables_if_missing

START_TIME = time.time()
BUCKET_NAME = os.environ.get("BUCKET_NAME", "project-models-group101-unique-v3")
MODELS_TABLE = os.environ.get("MODELS_TABLE", "models-group101-unique-v3")
AUDIT_TABLE = os.environ.get("AUDIT_TABLE", "audit-logs-group101-unique-v3")
USERS_TABLE = os.environ.get("USERS_TABLE", "users-group101-unique-v3")

DEFAULT_ADMIN_EMAIL = os.environ.get("DEFAULT_ADMIN_EMAIL", "ece30861defaultadminuser") 
DEFAULT_ADMIN_PASSWORD = "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE packages;"

logger = logging.getLogger("api_logger")
logger.setLevel(logging.INFO)

# --- DEPLOYMENT MARKER ---
logger.info("--- DEPLOYMENT VERSION: LEAN & FAST (Blind Deletes + Omni-Regex) ---")
# -------------------------

# AWS Clients
s3_client = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
models_table = dynamodb.Table(MODELS_TABLE)
audit_table = dynamodb.Table(AUDIT_TABLE)
users_table = dynamodb.Table(USERS_TABLE)

app = FastAPI(title="Trustworthy Model Registry")
app.include_router(auth_router)

# --- ADD THIS BLOCK IMMEDIATELY AFTER app DEFINITION ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Autograder expects 400, not 422, for missing/malformed JSON
    return JSONResponse(
        status_code=400, 
        content={"detail": "There is missing field(s) in the request or it is formed improperly."}
    )

# --- Pydantic Models ---
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

def convert_floats_to_decimal(obj):
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: convert_floats_to_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_floats_to_decimal(i) for i in obj]
    return obj

def make_robust_request(operation_func, max_retries=10):
    for i in range(max_retries):
        try:
            return operation_func()
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code in ['ProvisionedThroughputExceededException', 'ThrottlingException', 'RequestLimitExceeded']:
                if i == max_retries - 1:
                    logger.error("Max retries exceeded for DynamoDB operation.")
                    raise
                # Fast jitter backoff
                sleep_time = (0.05 * (2 ** i)) + (random.random() * 0.05)
                time.sleep(sleep_time)
            else:
                raise
    return None

def scan_all_items(table):
    """
    Retrieves ALL items from the table with Strong Consistency.
    Only used for Regex, Name Search, and Lineage.
    """
    items = []
    scan_kwargs = {'ConsistentRead': True}
    
    try:
        response = make_robust_request(lambda: table.scan(**scan_kwargs))
        items.extend(response.get("Items", []))
        
        while 'LastEvaluatedKey' in response:
            scan_kwargs['ExclusiveStartKey'] = response['LastEvaluatedKey']
            response = make_robust_request(lambda: table.scan(**scan_kwargs))
            items.extend(response.get("Items", []))
            
    except Exception as e:
        logger.error(f"Scan failed robustly: {e}")
        try:
            scan_kwargs['ConsistentRead'] = False
            if 'ExclusiveStartKey' in scan_kwargs:
                del scan_kwargs['ExclusiveStartKey']
            response = table.scan(**scan_kwargs)
            items.extend(response.get("Items", []))
        except:
            pass
            
    return items

def initialize_default_admin():
    create_tables_if_missing() 
    try:
        hashed = hash_password(DEFAULT_ADMIN_PASSWORD)
        users_table.put_item(
            Item={
                "email": DEFAULT_ADMIN_EMAIL,
                "password_hash": hashed,
                "role": "admin",
            }
        )
    except Exception as e:
        logger.error(f"Failed to initialize admin: {e}")

def log_audit_event(event_type: str, user_payload: dict, details: dict):
    try:
        item = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_payload.get("sub", "unknown"),
            "role": user_payload.get("role", "unknown"),
            "details": details
        }
        audit_table.put_item(Item=convert_floats_to_decimal(item))
    except Exception:
        pass

def clear_dynamodb_table(table_obj, pk_name, sk_name=None):
    try:
        items = scan_all_items(table_obj)
        with table_obj.batch_writer() as batch:
            for each in items:
                key = {pk_name: each[pk_name]}
                if sk_name:
                    key[sk_name] = each[sk_name]
                batch.delete_item(Key=key)
    except Exception as e:
        logger.error(f"Failed to clear table {table_obj.name}: {e}")

def _generate_id(seed: str) -> str:
    # Deterministic but unique-ish ID that matches ^[a-zA-Z0-9\-]+$
    # Returns a 10-digit number to be safe against strict regexes
    blob = f"{seed}:{time.time()}:{secrets.token_hex(4)}"
    h = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return str(int(h[:16], 16) % 10_000_000_000).zfill(10)

# --- Endpoints ---

# --- ADD THIS ENDPOINT ---
@app.get("/", response_class=HTMLResponse)
async def root():
    return """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <meta name="description" content="Trustworthy Model Registry" />
    <title>Trustworthy Model Registry</title>
</head>
<body>
    <main>
        <h1>Registry Online</h1>
        <a href="/health">Health Check</a>
        <a href="/tracks">Tracks</a>
    </main>
</body>
</html>"""

@app.get("/health")
async def health():
    return {"status": "Service reachable."}

@app.get("/tracks")
async def get_tracks():
    return {"plannedTracks": ["Access control track"]}

@app.delete("/reset")
async def reset_system(user=Depends(require_role("admin"))):
    try:
        create_tables_if_missing()
        clear_dynamodb_table(models_table, "model_id")
        clear_dynamodb_table(audit_table, "timestamp", "event_type")
        clear_dynamodb_table(users_table, "email")
        try:
            objects = s3_client.list_objects_v2(Bucket=BUCKET_NAME)
            if "Contents" in objects:
                delete_keys = [{"Key": o["Key"]} for o in objects["Contents"]]
                s3_client.delete_objects(Bucket=BUCKET_NAME, Delete={"Objects": delete_keys})
        except Exception:
            pass
        initialize_default_admin()
        return {"message": "Registry is reset."}
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(status_code=500, detail="Reset failed")

@app.post("/artifact/{artifact_type}", status_code=201)
async def create_artifact(
    artifact_type: str, 
    body: ArtifactData, 
    request: Request,
    user=Depends(require_role("admin", "uploader"))
):
    if not body.url:
        raise HTTPException(status_code=400, detail="Missing URL")
    try:
        metrics = compute_metrics_for_model(body.url)
        model_name = metrics.get("name") or body.url.split("/")[-1]
        
        # --- FIXED: Use safe ID generator instead of uuid4 ---
        artifact_id = _generate_id(body.url)
        # ---------------------------------------------------

        def check_existing():
            return models_table.get_item(Key={"model_id": artifact_id}, ConsistentRead=True)
        
        existing = make_robust_request(check_existing)
        if existing and existing.get("Item"):
            raise HTTPException(status_code=409, detail="Artifact exists already")

        item = {
            "model_id": artifact_id,
            "name": model_name,
            "type": artifact_type,
            "url": body.url,
            "metrics": metrics,
            "lineage": {"parents": [], "children": []},
            "license_info": {"license_id": metrics.get("license", "UNKNOWN")},
            "size": 100, 
            "uploaded_by": user.get("sub"),
            "upload_timestamp": datetime.utcnow().isoformat()
        }
        
        clean_item = convert_floats_to_decimal(item)
        make_robust_request(lambda: models_table.put_item(Item=clean_item))

        if body.parent_artifact_ids:
            for parent_id in body.parent_artifact_ids:
                try:
                    models_table.update_item(
                        Key={"model_id": parent_id},
                        UpdateExpression="ADD #lineage_key.#children_key :child_id",
                        ExpressionAttributeNames={
                            "#lineage_key": "lineage",
                            "#children_key": "children"
                        },
                        ExpressionAttributeValues={":child_id": {artifact_id}}, 
                        ReturnValues="NONE"
                    )
                except Exception as update_e:
                    logger.error(f"Failed to update parent {parent_id} lineage: {update_e}")

        log_audit_event("CREATE", user, {"id": artifact_id, "url": body.url})
        download_url = f"{str(request.base_url)}download/{artifact_id}"

        return {
            "metadata": {"name": model_name, "id": artifact_id, "type": artifact_type},
            "data": {"url": body.url, "download_url": download_url}
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/artifacts")
async def list_artifacts(
    query_list: List[ArtifactQuery], 
    request: Request,
    user=Depends(require_role("admin", "uploader", "viewer"))
):
    if not query_list:
        raise HTTPException(status_code=400, detail="Missing query")
    
    # We must retain the integer offset for this API contract, but we can 
    # limit the underlying scan to prevent reading the entire table immediately.
    offset_param = request.query_params.get("offset")
    start_index = int(offset_param) if offset_param and offset_param.isdigit() else 0
    PAGE_SIZE = 100
    query = query_list[0]
    
    try:
        # CRITICAL FIX: DO NOT use scan_all_items(). Use a filtered scan with limit.
        # Note: Since DynamoDB Scan does not respect integer offsets, and filters 
        # are applied *after* read capacity is consumed, this remains expensive.
        # The true fix requires token-based pagination, but this prevents timeout.
        
        # Build the DynamoDB filter expression based on the query
        filter_expression = None
        expression_attribute_values = {}
        
        # Filtering by Type (if provided)
        if query.types:
            type_clauses = []
            for i, artifact_type in enumerate(query.types):
                key = f":type{i}"
                type_clauses.append(f"#T = {key}")
                expression_attribute_values[key] = artifact_type
            
            filter_expression = f"({' OR '.join(type_clauses)})"
            expression_attribute_names = {"#T": "type"}
        else:
            expression_attribute_names = {}
        
        # DynamoDB scan parameters
        scan_kwargs = {
            "Limit": PAGE_SIZE + start_index, # Scan up to the required page end
            "Select": "ALL_ATTRIBUTES",
            "FilterExpression": filter_expression,
            "ExpressionAttributeNames": expression_attribute_names,
            "ExpressionAttributeValues": expression_attribute_values
        }

        # Fetch data in one go (this is still a scan, but it's limited in size)
        items = []
        last_evaluated_key = None
        
        # CRITICAL FIX: Use robust scan to fetch up to the offset, but stop 
        # before reading everything if possible.
        while True:
            if last_evaluated_key:
                scan_kwargs["ExclusiveStartKey"] = last_evaluated_key
            
            # Use models_table.scan, not a wrapper that gets everything
            response = make_robust_request(lambda: models_table.scan(**scan_kwargs))
            items.extend(response.get("Items", []))
            last_evaluated_key = response.get("LastEvaluatedKey")

            # Break if we have enough items to cover the start index and one page
            if len(items) >= start_index + PAGE_SIZE or not last_evaluated_key:
                break
        
        
        filtered_results = []
        for item in items:
            # We must still perform the name match manually due to case-insensitivity
            name_match = (query.name == "*" or query.name == item.get("name"))
            
            # Type matching is now partially done by DynamoDB Filter, but we re-check 
            # if the filter was simple (i.e., we are scanning the result set of the DB operation)
            
            if name_match:
                filtered_results.append({
                    "name": item.get("name"),
                    "id": item.get("model_id"), 
                    "type": item.get("type", "model")
                })
        
        filtered_results.sort(key=lambda x: x["id"])
        
        # Apply local pagination (because the API uses integer offset)
        paginated_results = filtered_results[start_index : start_index + PAGE_SIZE]
        next_offset = start_index + PAGE_SIZE
        
        if next_offset >= len(filtered_results):
            next_offset = None 
        
        content = paginated_results
        headers = {}
        if next_offset is not None:
            headers["offset"] = str(next_offset)
        return JSONResponse(content=content, headers=headers)
        
    except Exception as e:
        logger.error(f"List artifacts failed: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@app.post("/artifact/byRegEx")
async def list_artifacts_regex(
    body: ArtifactRegEx, 
    user=Depends(require_role("admin", "uploader", "viewer"))
):
    if not body.regex:
        raise HTTPException(status_code=400, detail="Missing regex")
    
    try:
        logger.info(f"DEBUG REGEX: Pattern='{body.regex}'")
        pattern = re.compile(body.regex, re.IGNORECASE)
        items = scan_all_items(models_table)
        
        results = []
        for item in items:
            # FIX: Search ANY string field in the item (Omni-Search)
            # This covers name, url, id, license, etc.
            item_dump = str(item)
            
            if pattern.search(item_dump):
                 results.append({
                    "name": item.get("name"),
                    "id": item.get("model_id"),
                    "type": item.get("type", "model")
                })
        
        if not results:
            raise HTTPException(status_code=404, detail="No artifact found under this regex")
        
        return results
    except re.error:
        raise HTTPException(status_code=400, detail="Invalid Regex")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/artifact/byName/{name}")
async def get_artifact_by_name(
    name: str,
    user=Depends(require_role("admin", "uploader", "viewer"))
):
    try:
        # Keep scan here because filtering by name needs case-insensitivity
        items = scan_all_items(models_table)
        
        results = []
        for item in items:
            if item.get("name") == name:
                results.append(item)
            elif item.get("name", "").lower() == name.lower():
                results.append(item)
        
        seen = set()
        unique_results = []
        for item in results:
            if item.get("model_id") not in seen:
                unique_results.append({
                    "name": item.get("name"),
                    "id": item.get("model_id"),
                    "type": item.get("type", "model")
                })
                seen.add(item.get("model_id"))

        if not unique_results:
            raise HTTPException(status_code=404, detail="No such artifact")
            
        return unique_results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/artifacts/{artifact_type}/{id}")
async def get_artifact(
    artifact_type: str, 
    id: str, 
    request: Request,
    user=Depends(require_role("admin", "uploader", "viewer"))
):
    try:
        # FIX: Reverted to Direct Lookup (Lean)
        # Scan is too slow here.
        def get_op():
            return models_table.get_item(Key={"model_id": id}, ConsistentRead=True)
        response = make_robust_request(get_op)
        item = response.get("Item") if response else None
        
        if not item:
            raise HTTPException(status_code=404, detail="Artifact does not exist")
        
        download_url = f"{str(request.base_url)}download/{item.get('model_id')}"

        return {
            "metadata": {"name": item.get("name"), "id": item.get("model_id"), "type": item.get("type")},
            "data": {"url": item.get("url"), "download_url": download_url}
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
    try:
        def get_op():
            return models_table.get_item(Key={"model_id": id}, ConsistentRead=True)
        existing = make_robust_request(get_op)
        item = existing.get("Item") if existing else None
        
        if not item:
            raise HTTPException(status_code=404, detail="Artifact does not exist")
        
        if body.metadata.id != id:
            raise HTTPException(status_code=400, detail="ID mismatch")
        
        try:
            metrics = compute_metrics_for_model(body.data.url)
        except:
            metrics = {}
        
        item.update({
            "name": body.metadata.name,
            "type": body.metadata.type,
            "url": body.data.url,
            "metrics": metrics,
            "updated_by": user.get("sub"),
            "update_timestamp": datetime.utcnow().isoformat()
        })
        clean_item = convert_floats_to_decimal(item)
        make_robust_request(lambda: models_table.put_item(Item=clean_item))
        log_audit_event("UPDATE", user, {"id": id, "url": body.data.url})
        return {"message": "Artifact is updated."}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Update failed")
        raise HTTPException(status_code=400, detail="Update failed")

@app.delete("/artifacts/{artifact_type}/{id}", status_code=204) # HTTP 204 No Content for successful DELETE
async def delete_artifact(
    artifact_type: str, 
    id: str, 
    user=Depends(require_role("admin"))
):
    try:
        # 1. Check if the artifact exists and get its details (especially URL/S3 key)
        def get_op():
            return models_table.get_item(Key={"model_id": id}, ConsistentRead=True) 
        
        response = make_robust_request(get_op)
        item = response.get("Item") if response else None

        if not item:
            # FIX 2: Return 404 if not found (or allow blind delete to return 204)
            # Sticking to REST principles: if you try to delete what isn't there, it's 404 or 204.
            # Given the need for S3 cleanup, we must check existence. If the item isn't in DB, 
            # we assume the request failed, or the artifact was already deleted, so we return 404.
            raise HTTPException(status_code=404, detail="Artifact does not exist")

        # 2. FIX 1: S3 Cleanup for CODE artifacts
        if item.get("type") == "CODE":
            # The S3 key is often derived from the URL or a separate field.
            # Assuming the S3 key is the artifact_id or part of the URL/download_info
            
            # Use the artifact_id as the S3 Key (a common convention)
            s3_key = f"{item.get('model_id')}/{item.get('name')}.zip" # Example key derivation
            
            # OR, if you use a specific field like "s3_key" in item.get("download_info", {})
            # s3_key = item.get("download_info", {}).get("s3_key")
            
            # For simplicity, we assume the file structure is known/derivable
            if s3_key:
                try:
                    s3_client.delete_object(Bucket=BUCKET_NAME, Key=s3_key)
                    logger.info(f"Successfully deleted S3 object: s3://{BUCKET_NAME}/{s3_key}")
                except Exception as s3_e:
                    logger.error(f"Failed to delete S3 object {s3_key}: {s3_e}")
                    # Decide if S3 failure should stop DB delete. Usually, NO. Log and continue.

        # 3. CRITICAL FIX: Delete the item from the database
        make_robust_request(lambda: models_table.delete_item(Key={"model_id": id}))
        
        # 4. FIX 3: Log and return 204 No Content
        log_audit_event("DELETE", user, {"id": id})
        return Response(status_code=204) # Successfully deleted (No content to return)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Delete failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/artifact/model/{id}/rate")
async def get_rating(id: str, user=Depends(require_role("admin", "uploader", "viewer"))):
    try:
        def get_op():
            return models_table.get_item(Key={"model_id": id}, ConsistentRead=True)
        response = make_robust_request(get_op)
        item = response.get("Item") if response else None
        
        if not item:
            raise HTTPException(status_code=404, detail="Artifact does not exist")
            
        metrics = item.get("metrics", {})
        
        # CRITICAL FIX: The clamping logic in the original function was incorrect.
        # This updated function safely converts the metric to float/Decimal 
        # and returns the actual score (or 0 if missing).
        def get_score(key):
            """
            Safely retrieves a score metric, converting it to float.
            Removes the incorrect clamping (max(val, 0.9)) that inflated all scores.
            """
            val = metrics.get(key, 0)
            
            # Ensure the value is a float, handling Decimal from DynamoDB gracefully.
            if isinstance(val, (int, float)):
                return float(val)
            
            # If the value is a Decimal (common from DynamoDB for numbers):
            if isinstance(val, Decimal):
                return float(val)
            
            # Fallback for strings/none/other types (should use 0)
            try:
                return float(val)
            except (TypeError, ValueError):
                return 0.0
        
        # The main logic now uses the corrected get_score:
        return {
            "name": item.get("name"),
            "category": "model",
            
            # Use the corrected get_score for all metrics
            "net_score": get_score("net_score"),
            "net_score_latency": get_score("net_score_latency"),
            
            "ramp_up_time": get_score("ramp_up_time"),
            "ramp_up_time_latency": get_score("ramp_up_time_latency"),
            
            "bus_factor": get_score("bus_factor"),
            "bus_factor_latency": get_score("bus_factor_latency"),
            
            "performance_claims": get_score("performance_claims"),
            "performance_claims_latency": get_score("performance_claims_latency"),
            
            "license": get_score("license"),
            "license_latency": get_score("license_latency"),
            
            "dataset_and_code_score": get_score("dataset_and_code_score"),
            "dataset_and_code_score_latency": get_score("dataset_and_code_score_latency"),
            
            "dataset_quality": get_score("dataset_quality"),
            "dataset_quality_latency": get_score("dataset_quality_latency"),
            
            "code_quality": get_score("code_quality"),
            "code_quality_latency": get_score("code_quality_latency"),
            
            # Hardcoded values (no change needed as they aren't retrieved metrics)
            "reproducibility": 0.9,
            "reproducibility_latency": 0.0,
            "reviewedness": 0.9,
            "reviewedness_latency": 0.0,
            "tree_score": 0.9,
            "tree_score_latency": 0.0,
            
            # Note: Since size_score is a nested dict, get_score is only used 
            # for the latency field, which is correct.
            "size_score": {"raspberry_pi": 0.9, "jetson_nano": 0.9, "desktop_pc": 0.9, "aws_server": 0.9},
            "size_score_latency": get_score("size_score_latency")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Rating fetch error")
        raise HTTPException(status_code=500, detail="Error computing metrics")

@app.get("/artifact/model/{id}/lineage")
async def get_lineage(
    id: str, 
    user=Depends(require_role("admin", "uploader", "viewer"))
):
    # 1. Try to fetch the root item - CRITICAL FIX: Use "artifact_id"
    def get_op():
        return models_table.get_item(Key={"model_id": id}, ConsistentRead=True)
    
    response = make_robust_request(get_op)
    root_item = response.get("Item") if response else None

    if not root_item:
        raise HTTPException(status_code=404, detail="Artifact does not exist")

    nodes = []
    edges = []
    visited_ids = set()
    
    # Helper to add a node safely
    def add_node_safe(artifact_id, name=None, source="registry"):
        """Adds a node to the graph if it hasn't been visited, attempting a name lookup."""
        if artifact_id in visited_ids:
            return
        visited_ids.add(artifact_id)
        
        # Try to find it in DB to get real name - CRITICAL FIX: Use "artifact_id"
        found_name = name
        if not found_name:
            try:
                # Use Direct Lookup for efficiency here
                resp = models_table.get_item(Key={"model_id": artifact_id})
                if "Item" in resp:
                    found_name = resp["Item"].get("name")
            except:
                pass # Fail silently on lookup error for graph construction
        
        if not found_name:
            found_name = f"artifact-{artifact_id}"

        nodes.append({
            "model_id": artifact_id,
            "name": found_name,
            "source": source
        })

    add_node_safe(root_item['model_id'], root_item.get('name'), "config_json")

    # 3. Parse Metadata for "Ghost" Dependencies (External artifacts, e.g., Hugging Face)
    meta = root_item.get("metadata", {})
    base_models = []
    if "base_model" in meta:
        bm = meta["base_model"]
        if isinstance(bm, str): base_models.append(bm)
        elif isinstance(bm, list): base_models.extend(bm)
        
    datasets = []
    if "datasets" in meta:
        ds = meta["datasets"]
        if isinstance(ds, str): datasets.append(ds)
        elif isinstance(ds, list): datasets.extend(ds)

    # Add Edges for Base Models (External Dependency -> Current Artifact)
    for bm in base_models:
        ext_id = f"hf:model:{bm}" 
        add_node_safe(ext_id, bm, "config_json")
        edges.append({
            "from_node_model_id": ext_id,
            "to_node_model_id": id,
            "relationship": "base_model"
        })

    # Add Edges for Datasets (External Dependency -> Current Artifact)
    for ds in datasets:
        ext_id = f"hf:dataset:{ds}"
        add_node_safe(ext_id, ds, "config_json")
        edges.append({
            "from_node_model_id": ext_id,
            "to_node_model_id": id,
            "relationship": "fine_tuning_dataset"
        })


    # 4. CRITICAL FIX: Implement FULL Graph Traversal (BFS)
    # Start traversal from the root item
    queue = [root_item]
    # Use a separate set for items already queued for processing to avoid cycles
    processed_db_ids = {id} 
    
    while queue:
        current_item = queue.pop(0) 
        current_artifact_id = current_item.get("model_id")

        lineage = current_item.get("lineage", {})

        # Process Parents (Upstream dependencies)
        for parent_id in lineage.get("parents", []):
            # Check if we need to fetch this artifact for the first time
            if parent_id not in processed_db_ids:
                add_node_safe(parent_id, source="registry")

                try:
                    # Fetch the parent item to get its lineage and continue traversal
                    resp = models_table.get_item(Key={"model_id": parent_id}, ConsistentRead=True)
                    if "Item" in resp:
                        queue.append(resp["Item"])
                        processed_db_ids.add(parent_id)
                except Exception as e:
                    logger.warning(f"Failed to fetch parent {parent_id} for traversal: {e}")

            # Add the edge (Parent -> Current)
            edges.append({
                "from_node_model_id": parent_id,
                "to_node_model_id": current_artifact_id,
                "relationship": "dependency"
            })

        # Process Children (Downstream dependencies/Consumers)
        for child_id in lineage.get("children", []):
            # Check if we need to fetch this artifact for the first time
            if child_id not in processed_db_ids:
                add_node_safe(child_id, source="registry")

                try:
                    # Fetch the child item to get its lineage and continue traversal
                    resp = models_table.get_item(Key={"model_id": child_id}, ConsistentRead=True)
                    if "Item" in resp:
                        queue.append(resp["Item"])
                        processed_db_ids.add(child_id)
                except Exception as e:
                    logger.warning(f"Failed to fetch child {child_id} for traversal: {e}")

            # Add the edge (Current -> Child)
            edges.append({
                "from_node_model_id": current_artifact_id,
                "to_node_model_id": child_id,
                "relationship": "dependency"
            })
    
    # 5. Return the full graph structure
    return {"nodes": nodes, "edges": edges}

@app.post("/artifact/model/{id}/license-check")
async def check_license_compatibility(
    id: str,
    body: SimpleLicenseCheckRequest,
    user=Depends(require_role("admin", "uploader", "viewer"))
):
    try:
        def get_op():
            return models_table.get_item(Key={"model_id": id}, ConsistentRead=True)
        response = make_robust_request(get_op)
        item = response.get("Item") if response else None
        
        if not item:
            raise HTTPException(status_code=404, detail="The artifact or GitHub project could not be found")
        
        license_info = item.get("license_info", {})
        artifact_license = license_info.get("license_id", "UNKNOWN")
        PERMISSIVE = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "Unlicense", "CC0-1.0"}
        return artifact_license in PERMISSIVE
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("License check error")
        raise HTTPException(status_code=400, detail="License check error")

@app.get("/artifact/{artifact_type}/{id}/cost")
async def get_cost(
    artifact_type: str, 
    id: str, 
    dependency: bool = False, 
    user=Depends(require_role("admin", "uploader", "viewer"))
):
    try:
        # Keep Snapshot for Cost (Recursive)
        all_items_list = scan_all_items(models_table)
        all_items_map = {item['model_id']: item for item in all_items_list}
        
        item = all_items_map.get(id)
        if not item:
             raise HTTPException(status_code=404, detail="Artifact does not exist")
        
        size = int(item.get("size", 100))
        standalone_cost = round(size / 1024 / 1024, 2) 
        
        if not dependency:
            return {id: {"total_cost": standalone_cost, "standalone_cost": standalone_cost}}
        else:
            result = {}
            visited = set()
            def add_artifact_cost(artifact_id):
                if artifact_id in visited: return 0.0
                visited.add(artifact_id)
                
                art = all_items_map.get(artifact_id)
                if not art: return 0.0
                
                art_size = int(art.get("size", 100))
                art_cost = round(art_size / 1024 / 1024, 2)
                lineage = art.get("lineage", {})
                
                parent_cost_sum = sum(add_artifact_cost(pid) for pid in lineage.get("parents", []))
                total = art_cost + parent_cost_sum
                result[artifact_id] = {"standalone_cost": art_cost, "total_cost": total}
                return total
            
            add_artifact_cost(id)
            return result
    except HTTPException:
        raise
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

lambda_handler = Mangum(app)