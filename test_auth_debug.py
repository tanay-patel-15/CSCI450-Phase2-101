#!/usr/bin/env python3
"""
Debug script to test authentication locally
"""
import requests
import json

# Test authentication endpoint
base_url = "http://localhost:8000"

# Exact credentials from spec
admin_email = "ece30861defaultadminuser"
admin_password = 'correcthorsebatterystaple123(!__+@**(A\'"`;DROP TABLE artifacts;'

print("=" * 60)
print("Testing Authentication Endpoint")
print("=" * 60)

# Test 1: POST /authenticate
print("\n1. Testing POST /authenticate")
print(f"   Email: {admin_email}")
print(f"   Password: {admin_password[:20]}...")

auth_payload = {
    "user": {
        "name": admin_email,
        "is_admin": True
    },
    "secret": {
        "password": admin_password
    }
}

print(f"\n   Payload: {json.dumps(auth_payload, indent=2)}")

try:
    response = requests.post(
        f"{base_url}/authenticate",
        json=auth_payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"\n   Status: {response.status_code}")
    print(f"   Response: {response.text}")
    
    if response.status_code == 200:
        print("\n   ✅ Authentication SUCCESS!")
    else:
        print("\n   ❌ Authentication FAILED!")
        
except Exception as e:
    print(f"\n   ❌ ERROR: {e}")

print("\n" + "=" * 60)
