import boto3
import os
import logging
import time

logger = logging.getLogger("db_setup")
logger.setLevel(logging.INFO)

def wait_for_table_active(client, table_name, max_wait=60):
    """Wait for a table to become ACTIVE."""
    start = time.time()
    while time.time() - start < max_wait:
        try:
            response = client.describe_table(TableName=table_name)
            status = response['Table']['TableStatus']
            if status == 'ACTIVE':
                logger.info(f"Table {table_name} is now ACTIVE")
                return True
            time.sleep(1)
        except client.exceptions.ResourceNotFoundException:
            # Table might have just been deleted, wait
            time.sleep(1)
        except Exception as e:
            logger.warning(f"Waiting for {table_name}: {e}")
            time.sleep(1)
    logger.warning(f"Table {table_name} did not become ACTIVE within {max_wait}s")
    return False

def wait_for_table_deleted(client, table_name, max_wait=60):
    """Wait for a table to disappear."""
    start = time.time()
    while time.time() - start < max_wait:
        try:
            client.describe_table(TableName=table_name)
            # If we can describe it, it still exists
            logger.info(f"Table {table_name} still exists, waiting for deletion...")
            time.sleep(1)
        except client.exceptions.ResourceNotFoundException:
            logger.info(f"Table {table_name} deleted successfully.")
            return True
        except Exception:
            time.sleep(1)
    return False

def validate_and_recreate_table(client, dynamodb, table_name, key_schema, attribute_definitions):
    """
    Checks if table exists. 
    If it exists but has the WRONG KeySchema, deletes and recreates it.
    If it doesn't exist, creates it.
    """
    try:
        response = client.describe_table(TableName=table_name)
        existing_keys = response['Table']['KeySchema']
        
        # Check if Schema matches
        # We only check if the AttributeNames match the KeySchema requirements
        required_keys = sorted([k['AttributeName'] for k in key_schema])
        current_keys = sorted([k['AttributeName'] for k in existing_keys])
        
        if required_keys == current_keys:
            logger.info(f"Table {table_name} exists with correct schema.")
            return False # No action taken
        
        logger.warning(f"Table {table_name} has WRONG schema (Expected {required_keys}, got {current_keys}). Deleting...")
        
        # Delete invalid table
        client.delete_table(TableName=table_name)
        wait_for_table_deleted(client, table_name)
        
    except client.exceptions.ResourceNotFoundException:
        # Table does not exist, proceed to creation
        pass
        
    # Create Table
    logger.info(f"Creating table: {table_name}")
    try:
        dynamodb.create_table(
            TableName=table_name,
            KeySchema=key_schema,
            AttributeDefinitions=attribute_definitions,
            ProvisionedThroughput={'ReadCapacityUnits': 5, 'WriteCapacityUnits': 5}
        )
        return True # Created
    except Exception as e:
        logger.error(f"Failed to create {table_name}: {e}")
        return False

def create_tables_if_missing():
    """
    Ensures tables exist AND have the correct schema.
    """
    MODELS_TABLE = os.environ.get("MODELS_TABLE", "models")
    USERS_TABLE = os.environ.get("USERS_TABLE", "users")
    AUDIT_TABLE = os.environ.get("AUDIT_TABLE", "audit_logs")

    dynamodb = boto3.resource("dynamodb")
    client = boto3.client("dynamodb")
    
    tables_to_wait_for = []

    # --- Users Table (Target of the error) ---
    # Expected Key: email (HASH)
    created = validate_and_recreate_table(
        client, dynamodb, USERS_TABLE,
        key_schema=[{'AttributeName': 'email', 'KeyType': 'HASH'}],
        attribute_definitions=[{'AttributeName': 'email', 'AttributeType': 'S'}]
    )
    if created: tables_to_wait_for.append(USERS_TABLE)

    # --- Models Table ---
    # Expected Key: model_id (HASH)
    created = validate_and_recreate_table(
        client, dynamodb, MODELS_TABLE,
        key_schema=[{'AttributeName': 'model_id', 'KeyType': 'HASH'}],
        attribute_definitions=[{'AttributeName': 'model_id', 'AttributeType': 'S'}]
    )
    if created: tables_to_wait_for.append(MODELS_TABLE)

    # --- Audit Log Table ---
    # Expected Key: timestamp (HASH), event_type (RANGE)
    created = validate_and_recreate_table(
        client, dynamodb, AUDIT_TABLE,
        key_schema=[
            {'AttributeName': 'timestamp', 'KeyType': 'HASH'},
            {'AttributeName': 'event_type', 'KeyType': 'RANGE'}
        ],
        attribute_definitions=[
            {'AttributeName': 'timestamp', 'AttributeType': 'S'},
            {'AttributeName': 'event_type', 'AttributeType': 'S'}
        ]
    )
    if created: tables_to_wait_for.append(AUDIT_TABLE)

    # Wait for any newly created/recreated tables
    for table in tables_to_wait_for:
        wait_for_table_active(client, table)
