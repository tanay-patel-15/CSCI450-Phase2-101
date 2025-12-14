#!/usr/bin/env python3
"""
Test password escaping - make sure both files have the EXACT same password string
"""

print("Checking password consistency between files:")
print("=" * 70)

# From auth.py line 16
auth_py_password = 'correcthorsebatterystaple123(!__+@**(A\'"`;DROP TABLE artifacts;'

# From api.py line 27  
api_py_password = 'correcthorsebatterystaple123(!__+@**(A\'"`;DROP TABLE artifacts;'

print(f"\nauth.py password: {repr(auth_py_password)}")
print(f"api.py password:  {repr(api_py_password)}")

if auth_py_password == api_py_password:
    print("\n✅ Passwords MATCH!")
else:
    print("\n❌ Passwords DO NOT MATCH!")
    print("\nDifferences:")
    for i, (c1, c2) in enumerate(zip(auth_py_password, api_py_password)):
        if c1 != c2:
            print(f"  Position {i}: auth.py has {repr(c1)}, api.py has {repr(c2)}")
    
    if len(auth_py_password) != len(api_py_password):
        print(f"\nLength difference: auth.py={len(auth_py_password)}, api.py={len(api_py_password)}")

print("\n" + "=" * 70)
print(f"\nExpected final password:\n{auth_py_password}")
print(f"\nLength: {len(auth_py_password)} characters")
print(f"Repr: {repr(auth_py_password)}")
