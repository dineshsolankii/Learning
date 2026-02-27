import requests

# =============================================================================
# HTTP PATCH REQUEST
# =============================================================================
# PURPOSE : PARTIALLY UPDATE a resource — only the fields you send get changed.
#           Everything else stays exactly as it was.
# SAFE    : No — it modifies server state.
# IDEMPOTENT: Debatable — generally yes if same patch applied multiple times.
# BODY    : Yes — only send the fields you want to change.
#
# REAL-WORLD USE CASES:
#   - Change only a user's email (don't touch name, phone, etc.)
#   - Mark an order as "shipped" without changing other order details
#   - Toggle a feature flag
#
# PATCH vs PUT (key interview question!)
#   PATCH = send only changed fields → safe for partial updates
#   PUT   = send ALL fields          → replaces entire resource
# =============================================================================

BASE_URL = "https://jsonplaceholder.typicode.com"

# ── 1. Basic PATCH — update only the title ───────────────────────────────────
print("=== 1. Update only the title (PATCH /posts/1) ===")

# We only send the field we want to change
# The 'body' and 'userId' on the server remain untouched
partial_update = {
    "title": "New Title — Only This Changed"
}

response = requests.patch(f"{BASE_URL}/posts/1", json=partial_update)

print("Status Code   :", response.status_code)  # 200 = OK
print("Updated Post  :", response.json())        # Only title changed; body still there

# ── 2. PATCH multiple fields at once ─────────────────────────────────────────
print("\n=== 2. Update multiple fields at once ===")

multi_update = {
    "title": "Updated Title",
    "body" : "Updated body content — userId stays untouched on the server"
}

response = requests.patch(f"{BASE_URL}/posts/1", json=multi_update)

print("Status Code :", response.status_code)
print("Response    :", response.json())

# ── 3. PATCH with Authorization header ───────────────────────────────────────
print("\n=== 3. PATCH with Authorization header ===")

headers = {
    "Content-Type" : "application/json",
    "Authorization": "Bearer my-secret-token"
}

# Example: mark a todo as completed (without changing the title)
status_update = {"completed": True}

response = requests.patch(f"{BASE_URL}/todos/1", json=status_update, headers=headers)

print("Status Code :", response.status_code)
print("Response    :", response.json())

# ── 4. Practical comparison: PATCH vs PUT ────────────────────────────────────
print("\n=== 4. PATCH vs PUT side-by-side ===")

# Imagine the server has: {"id": 1, "name": "Alice", "email": "alice@old.com", "age": 30}

# PATCH — only email changes, name and age stay intact
patch_payload = {"email": "alice@new.com"}
print("PATCH payload (only email):", patch_payload)
print("Server result → name='Alice', email='alice@new.com', age=30  ✓")

# PUT — you MUST send all fields or others get wiped
put_payload = {"id": 1, "name": "Alice", "email": "alice@new.com", "age": 30}
print("\nPUT payload (all fields)  :", put_payload)
print("Server result → all fields replaced  (risky if you forget a field!)")

# ── Key things to know ────────────────────────────────────────────────────────
# Only send fields you want to change in PATCH — others are untouched.
# PATCH is the safer choice when doing partial updates.
# 200 OK or 204 No Content are typical success codes.
# Some APIs only support PUT (not PATCH) — check docs before using.
