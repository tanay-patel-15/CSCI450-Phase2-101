from jose import jwt

SECRET = "test_secret"

def make_jwt(role="viewer"):
    return jwt.encode({"sub": "test@user.com", "role": role}, SECRET)