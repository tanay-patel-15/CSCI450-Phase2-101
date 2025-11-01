import boto3
from botocore.exceptions import ClientError

s3_client = boto3.client('s3', region_name='us-east-1')
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
models_table = dynamodb.Table('Models')
BUCKET_NAME = "model-registry-artifacts"

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    
    # Upload to S3
    s3_client.put_object(Bucket=BUCKET_NAME, Key=file.filename, Body=content)
    
    # Store metadata in DynamoDB
    models_table.put_item(Item={
        "model_name": file.filename,
        "size": len(content),
        "status": "uploaded"
    })
    
    return {"filename": file.filename, "size": len(content)}


