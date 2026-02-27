import requests

# =============================================================================
# HTTP POST REQUEST
# =============================================================================
# PURPOSE : CREATE a new resource on the server.
# SAFE    : No — it modifies server state (creates data).
# IDEMPOTENT: No — sending POST twice creates TWO separate records.
# BODY    : Yes — data sent in the request body (JSON, form-data, etc.)
#
# REAL-WORLD USE CASES:
#   - User sign-up (POST /register)
#   - Submit a comment / order
#   - Upload a file
# =============================================================================

BASE_URL = "https://jsonplaceholder.typicode.com"

# ── 1. POST with JSON body ────────────────────────────────────────────────────
print("=== 1. Create a new post (POST /posts) ===")

# The data you want to send to the server
new_post = {
    "title"  : "Hello World",
    "body"   : "This is my first post via API",
    "userId" : 1
}

# json= automatically sets Content-Type: application/json header
response = requests.post(f"{BASE_URL}/posts", json=new_post)

print("Status Code :", response.status_code)  # 201 = Created (new resource made)
print("Created Post:", response.json())        # Server echoes back the new object with its ID

# ── 2. POST with custom headers ───────────────────────────────────────────────
print("\n=== 2. POST with Authorization header ===")

headers = {
    "Content-Type" : "application/json",     # We're sending JSON
    "Authorization": "Bearer my-secret-token" # Authenticate the request
}

payload = {"title": "Authenticated Post", "body": "Secure content", "userId": 2}

response = requests.post(f"{BASE_URL}/posts", json=payload, headers=headers)

print("Status Code :", response.status_code)
print("Response    :", response.json())

# ── 3. POST with form-encoded data (like an HTML form) ───────────────────────
print("\n=== 3. POST with form data ===")

# data= sends as application/x-www-form-urlencoded (classic HTML form format)
form_data = {"username": "dinesh", "password": "secret123"}
response = requests.post("https://httpbin.org/post", data=form_data)

print("Status Code :", response.status_code)
print("Form data received by server:", response.json().get("form"))

# ── Key things to know ────────────────────────────────────────────────────────
# json=  → sends data as JSON, sets Content-Type: application/json automatically
# data=  → sends as form-encoded (key=value&key=value)
# 201 Created is the ideal status for a successful POST
# The server typically returns the new object including its auto-generated ID
