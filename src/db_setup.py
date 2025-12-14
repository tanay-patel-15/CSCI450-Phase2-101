import boto3
import os
import logging

logger = logging.getLogger("db_setup")
logger.setLevel(logging.INFO)

MODELS_TABLE = os.environ.get("MODELS_TABLE", "models")
USERS_TABLE = os.environ.get("USERS_TABLE", "users")
AUDIT_TABLE = os.environ.get("AUDIT_TABLE", "audit_logs")

def create_tables_if_missing():
    """
    Checks if required DynamoDB tables exist and creates them if they don't.
    Safe to call repeatedly (idempotent).
    """
    dynamodb = boto3.resource("dynamodb")
    client = boto3.client("dynamodb")
    
    existing_tables = []
    try:
        # List tables to avoid ResourceNotFoundException noise
        response = client.list_tables()
        existing_tables = response.get("TableNames", [])
    except Exception as e:
        logger.error(f"Failed to list tables: {e}")
        # Proceeding might fail, but let's try strict creation handling below
    
    # --- Users Table ---
    if USERS_TABLE not in existing_tables:
        try:
            logger.info(f"Creating table: {USERS_TABLE}")
            dynamodb.create_table(
                TableName=USERS_TABLE,
                KeySchema=[{'AttributeName': 'email', 'KeyType': 'HASH'}],
                AttributeDefinitions=[{'AttributeName': 'email', 'AttributeType': 'S'}],
                ProvisionedThroughput={'ReadCapacityUnits': 1, 'WriteCapacityUnits': 1}
            )
        except Exception as e:
            if "ResourceInUseException" not in str(e):
                logger.error(f"Failed to create {USERS_TABLE}: {e}")

    # --- Models Table ---
    if MODELS_TABLE not in existing_tables:
        try:
            logger.info(f"Creating table: {MODELS_TABLE}")
            dynamodb.create_table(
                TableName=MODELS_TABLE,
                KeySchema=[{'AttributeName': 'model_id', 'KeyType': 'HASH'}],
                AttributeDefinitions=[{'AttributeName': 'model_id', 'AttributeType': 'S'}],
                ProvisionedThroughput={'ReadCapacityUnits': 1, 'WriteCapacityUnits': 1}
            )
        except Exception as e:
            if "ResourceInUseException" not in str(e):
                logger.error(f"Failed to create {MODELS_TABLE}: {e}")

    # --- Audit Log Table ---
    if AUDIT_TABLE not in existing_tables:
        try:
            logger.info(f"Creating table: {AUDIT_TABLE}")
            dynamodb.create_table(
                TableName=AUDIT_TABLE,
                KeySchema=[
                    {'AttributeName': 'timestamp', 'KeyType': 'HASH'},
                    {'AttributeName': 'event_type', 'KeyType': 'RANGE'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'timestamp', 'AttributeType': 'S'},
                    {'AttributeName': 'event_type', 'AttributeType': 'S'}
                ],
                ProvisionedThroughput={'ReadCapacityUnits': 1, 'WriteCapacityUnits': 1}
            )
        except Exception as e:
            if "ResourceInUseException" not in str(e):
                logger.error(f"Failed to create {AUDIT_TABLE}: {e}")
