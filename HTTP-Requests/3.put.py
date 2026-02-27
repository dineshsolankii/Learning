import requests

# =============================================================================
# HTTP PUT REQUEST
# =============================================================================
# PURPOSE : REPLACE an entire resource with the new data you send.
#           If a field is missing from the body, it gets wiped out.
# SAFE    : No — it modifies server state.
# IDEMPOTENT: Yes — calling PUT 5 times with the same body = same final state.
# BODY    : Yes — must send the COMPLETE updated object.
#
# REAL-WORLD USE CASES:
#   - Update a user's entire profile (all fields replaced)
#   - Replace a configuration file
#   - Update product details (all fields)
#
# PUT vs PATCH (key interview question!)
#   PUT   = full replacement → you send ALL fields
#   PATCH = partial update   → you send only changed fields
# =============================================================================

BASE_URL = "https://jsonplaceholder.typicode.com"

# ── 1. Basic PUT — replace the entire post with id=1 ─────────────────────────
print("=== 1. Full replacement of post (PUT /posts/1) ===")

# You MUST send ALL fields — any missing field will be overwritten with nothing
updated_post = {
    "id"     : 1,         # Include the ID so server knows which record
    "title"  : "Updated Title",
    "body"   : "This is the completely updated body content",
    "userId" : 1
}

response = requests.put(f"{BASE_URL}/posts/1", json=updated_post)

print("Status Code  :", response.status_code)  # 200 = OK (resource updated)
print("Updated Post :", response.json())

# ── 2. Why missing fields are dangerous in PUT ───────────────────────────────
print("\n=== 2. Danger: Missing fields in PUT ===")

# WRONG — only sending title, body and userId will be wiped on a real server!
partial_update = {
    "title": "Only Title Sent"
    # body and userId are missing → they get deleted on a real server
}

response = requests.put(f"{BASE_URL}/posts/1", json=partial_update)

print("Status Code  :", response.status_code)
print("Result (body & userId may be gone on real server):", response.json())
print("NOTE: jsonplaceholder is a mock API so it won't actually wipe data.")

# ── 3. PUT with headers ───────────────────────────────────────────────────────
print("\n=== 3. PUT with Authorization header ===")

headers = {
    "Content-Type" : "application/json",
    "Authorization": "Bearer my-token"
}

complete_post = {
    "id"    : 2,
    "title" : "Fully Updated Post 2",
    "body"  : "All fields present — safe PUT",
    "userId": 1
}

response = requests.put(f"{BASE_URL}/posts/2", json=complete_post, headers=headers)

print("Status Code :", response.status_code)
print("Response    :", response.json())

# ── Key things to know ────────────────────────────────────────────────────────
# Always send ALL fields in PUT — missing fields get wiped.
# Use PATCH if you only want to update specific fields.
# 200 OK or 204 No Content are typical success codes for PUT.
# PUT to a non-existent resource → some APIs create it (upsert), others return 404.
