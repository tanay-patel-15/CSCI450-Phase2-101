def test_register_and_login(client, db_users_table):
    # Register
    r = client.post("/register", params={"email": "test@a.com", "password": "secret"})
    assert r.status_code == 200

    # Login
    r = client.post("/login", params={"email": "test@a.com", "password": "secret"})
    assert r.status_code == 200
    assert "access_token" in r.json()