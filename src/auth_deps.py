from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer
from jose import jwt
import os

auth_scheme = HTTPBearer()
SECRET = os.getenv("JWT_SECRET", "dev_secret")

def require_role(*roles):
    def wrapper(credentials=Depends(auth_scheme)):
        token = credentials.credentials
        try:
            payload = jwt.decode(token, SECRET, algorithms=["HS256"])
            if payload["role"] not in roles:
                raise HTTPException(403, "Forbidden")
            return payload
        except:
            raise HTTPException(401, "Invalid token")
    return wrapper