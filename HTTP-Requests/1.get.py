import requests

# =============================================================================
# HTTP GET REQUEST
# =============================================================================
# PURPOSE : Fetch / Read data from a server. Does NOT modify anything.
# SAFE    : Yes — GET never changes server state.
# IDEMPOTENT: Yes — calling it 100 times gives the same result as calling once.
# BODY    : No — data is passed via URL query params, not a request body.
#
# REAL-WORLD USE CASES:
#   - Load a user profile page
#   - Fetch a list of products
#   - Search results (GET /search?q=python)
# =============================================================================

BASE_URL = "https://jsonplaceholder.typicode.com"

# ── 1. Basic GET — fetch a single post by ID ─────────────────────────────────
print("=== 1. Fetch single post (GET /posts/1) ===")

response = requests.get(f"{BASE_URL}/posts/1")

# Status code tells you whether the request succeeded
print("Status Code :", response.status_code)   # 200 = OK

# .json() parses the JSON body into a Python dict
data = response.json()
print("Response    :", data)

# ── 2. GET with Query Parameters — filter/search ─────────────────────────────
print("\n=== 2. Filter posts by userId (GET /posts?userId=1) ===")

# Query params are key-value pairs appended to the URL: ?userId=1
params = {"userId": 1}
response = requests.get(f"{BASE_URL}/posts", params=params)

print("Status Code :", response.status_code)
posts = response.json()
print(f"Total posts for userId=1 : {len(posts)}")
print("First post  :", posts[0])

# ── 3. GET with Custom Headers ────────────────────────────────────────────────
print("\n=== 3. GET with custom headers ===")

# Headers carry metadata — e.g. Accept tells the server what format you want.
headers = {
    "Accept": "application/json",       # We want JSON back
    "Authorization": "Bearer fake-token" # Normally a real JWT/API key
}
response = requests.get(f"{BASE_URL}/posts/1", headers=headers)

print("Status Code :", response.status_code)
print("Response Headers:", dict(response.headers))  # Server's response headers
print("Body        :", response.json())

# ── Key things to know ────────────────────────────────────────────────────────
# response.status_code  → integer (200, 404, 500 …)
# response.json()       → parsed Python dict / list
# response.text         → raw string body
# response.headers      → dict of response headers
# response.url          → final URL that was requested
