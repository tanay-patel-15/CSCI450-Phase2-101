from fastapi import FastAPI, UploadFile, File
from metrics import compute_metrics_for_model  # relative import, src is PYTHONPATH

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
    # TODO: save to S3 + add metadata to DynamoDB
    return {"filename": file.filename, "size": len(content)}



