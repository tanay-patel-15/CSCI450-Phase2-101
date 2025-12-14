import boto3
import os
import logging
import time

logger = logging.getLogger("db_setup")
logger.setLevel(logging.INFO)

def wait_for_table_active(client, table_name, max_wait=30):
    """Wait for a table to become ACTIVE. Works with both real AWS and moto."""
    start = time.time()
    while time.time() - start < max_wait:
        try:
            response = client.describe_table(TableName=table_name)
            status = response['Table']['TableStatus']
            if status == 'ACTIVE':
                logger.info(f"Table {table_name} is now ACTIVE")
                return True
            logger.info(f"Table {table_name} status: {status}, waiting...")
            time.sleep(0.5)
        except Exception as e:
            # Table might not exist yet in list, brief wait
            time.sleep(0.5)
    logger.warning(f"Table {table_name} did not become ACTIVE within {max_wait}s")
    return False

def create_tables_if_missing():
    """
    Checks if required DynamoDB tables exist and creates them if they don't.
    Waits for tables to become ACTIVE before returning.
    Safe to call repeatedly (idempotent).
    """
    # Read env vars at runtime to ensure we pick up any changes (e.g. during tests)
    MODELS_TABLE = os.environ.get("MODELS_TABLE", "models")
    USERS_TABLE = os.environ.get("USERS_TABLE", "users")
    AUDIT_TABLE = os.environ.get("AUDIT_TABLE", "audit_logs")

    dynamodb = boto3.resource("dynamodb")
    client = boto3.client("dynamodb")
    
    existing_tables = []
    try:
        response = client.list_tables()
        existing_tables = response.get("TableNames", [])
    except Exception as e:
        logger.error(f"Failed to list tables: {e}")
    
    tables_created = []
    
    # --- Users Table ---
    if USERS_TABLE not in existing_tables:
        try:
            logger.info(f"Creating table: {USERS_TABLE}")
            dynamodb.create_table(
                TableName=USERS_TABLE,
                KeySchema=[{'AttributeName': 'email', 'KeyType': 'HASH'}],
                AttributeDefinitions=[{'AttributeName': 'email', 'AttributeType': 'S'}],
                ProvisionedThroughput={'ReadCapacityUnits': 5, 'WriteCapacityUnits': 5}
            )
            tables_created.append(USERS_TABLE)
        except client.exceptions.ResourceInUseException:
            logger.info(f"Table {USERS_TABLE} already exists (race condition)")
        except Exception as e:
            logger.error(f"Failed to create {USERS_TABLE}: {e}")

    # --- Models Table ---
    if MODELS_TABLE not in existing_tables:
        try:
            logger.info(f"Creating table: {MODELS_TABLE}")
            dynamodb.create_table(
                TableName=MODELS_TABLE,
                KeySchema=[{'AttributeName': 'model_id', 'KeyType': 'HASH'}],
                AttributeDefinitions=[{'AttributeName': 'model_id', 'AttributeType': 'S'}],
                ProvisionedThroughput={'ReadCapacityUnits': 5, 'WriteCapacityUnits': 5}
            )
            tables_created.append(MODELS_TABLE)
        except client.exceptions.ResourceInUseException:
            logger.info(f"Table {MODELS_TABLE} already exists (race condition)")
        except Exception as e:
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
                ProvisionedThroughput={'ReadCapacityUnits': 5, 'WriteCapacityUnits': 5}
            )
            tables_created.append(AUDIT_TABLE)
        except client.exceptions.ResourceInUseException:
            logger.info(f"Table {AUDIT_TABLE} already exists (race condition)")
        except Exception as e:
            logger.error(f"Failed to create {AUDIT_TABLE}: {e}")

    # Wait for ALL created tables to become ACTIVE
    for table_name in tables_created:
        wait_for_table_active(client, table_name)
