import requests

# =============================================================================
# HTTP DELETE REQUEST
# =============================================================================
# PURPOSE : DELETE / Remove a resource from the server permanently.
# SAFE    : No — it modifies server state (removes data).
# IDEMPOTENT: Yes — deleting something that's already gone = same result (gone).
#             (Server may return 404 on subsequent calls, but state is the same.)
# BODY    : No (usually) — the target is identified by the URL.
#
# REAL-WORLD USE CASES:
#   - Delete a user account (DELETE /users/42)
#   - Remove a comment or post
#   - Cancel / delete an order
#   - Revoke an API token
# =============================================================================

BASE_URL = "https://jsonplaceholder.typicode.com"

# ── 1. Basic DELETE — remove a post by ID ────────────────────────────────────
print("=== 1. Delete a post (DELETE /posts/1) ===")

response = requests.delete(f"{BASE_URL}/posts/1")

print("Status Code :", response.status_code)
# 200 = OK (with body confirming deletion)
# 204 = No Content (success but no body — most common for DELETE)
# 404 = Not Found (already deleted or never existed)

# jsonplaceholder returns 200 with an empty object {}
print("Response    :", response.json())  # {} — empty body means it's gone

# ── 2. DELETE with Authorization header ──────────────────────────────────────
print("\n=== 2. DELETE with Authorization (protected resource) ===")

# Most real APIs require you to be authenticated before deleting
headers = {
    "Authorization": "Bearer my-secret-token"
}

response = requests.delete(f"{BASE_URL}/posts/5", headers=headers)

print("Status Code :", response.status_code)
print("Response    :", response.text)  # Often empty body on real APIs

# ── 3. Handle response codes properly ────────────────────────────────────────
print("\n=== 3. Handling DELETE response codes ===")

post_id = 10
response = requests.delete(f"{BASE_URL}/posts/{post_id}")

if response.status_code == 200:
    print(f"Post {post_id} deleted successfully (200 OK)")
elif response.status_code == 204:
    print(f"Post {post_id} deleted successfully (204 No Content)")
elif response.status_code == 404:
    print(f"Post {post_id} not found — may already be deleted (404)")
elif response.status_code == 403:
    print("Permission denied — you are not allowed to delete this (403 Forbidden)")
else:
    print(f"Unexpected status: {response.status_code}")

# ── 4. Idempotency explained ──────────────────────────────────────────────────
print("\n=== 4. DELETE is idempotent — safe to retry ===")

# First delete
r1 = requests.delete(f"{BASE_URL}/posts/3")
print(f"First  DELETE → {r1.status_code}")  # 200

# Second delete (resource already gone on a real server → 404, but state is same: deleted)
r2 = requests.delete(f"{BASE_URL}/posts/3")
print(f"Second DELETE → {r2.status_code}")  # 200 on mock; 404 on real server
print("Either way, the resource is gone — state is identical both times.")

# ── Key things to know ────────────────────────────────────────────────────────
# 200 → Deleted, body confirms it
# 204 → Deleted, no body (most RESTful APIs use this)
# 404 → Resource not found (idempotent — still "gone")
# 403 → Not authorized to delete
# Always confirm with the user before deleting — it's irreversible!
