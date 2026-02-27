import requests

# =============================================================================
# HTTP HEAD REQUEST
# =============================================================================
# PURPOSE : Same as GET but the server returns ONLY the headers — no body.
#           Use it when you need metadata about a resource without downloading it.
# SAFE    : Yes — never modifies anything.
# IDEMPOTENT: Yes — same result every time.
# BODY    : No response body (that's the whole point!).
#
# REAL-WORLD USE CASES:
#   - Check if a resource exists (without downloading it)
#   - Get file size before downloading (Content-Length header)
#   - Check if a cached resource is still fresh (Last-Modified / ETag)
#   - Validate a URL (is it alive? what content type is it?)
#   - Health checks in load balancers / monitoring tools
# =============================================================================

BASE_URL = "https://jsonplaceholder.typicode.com"

# ── 1. Basic HEAD — check if a resource exists ───────────────────────────────
print("=== 1. Check if a post exists (HEAD /posts/1) ===")

response = requests.head(f"{BASE_URL}/posts/1")

print("Status Code  :", response.status_code)   # 200 = exists, 404 = not found
print("Body         :", repr(response.text))     # Always empty — no body in HEAD
print("Content-Type :", response.headers.get("Content-Type"))

# ── 2. Get file/resource metadata without downloading ────────────────────────
print("\n=== 2. Peek at resource metadata ===")

response = requests.head(f"{BASE_URL}/posts")

print("Status Code    :", response.status_code)
print("Content-Type   :", response.headers.get("Content-Type"))
print("Content-Length :", response.headers.get("Content-Length", "Not provided"))
# Content-Length tells you file size in bytes BEFORE downloading — very useful!

# ── 3. Check freshness — has the resource changed since last time? ────────────
print("\n=== 3. Cache validation headers ===")

response = requests.head(f"{BASE_URL}/posts/1")

# ETag — unique identifier for the current version of the resource
etag = response.headers.get("ETag", "Not provided")

# Last-Modified — when the resource was last changed
last_modified = response.headers.get("Last-Modified", "Not provided")

print("ETag          :", etag)
print("Last-Modified :", last_modified)
print("If these match your cached version → no need to re-download!")

# ── 4. Compare HEAD vs GET ────────────────────────────────────────────────────
print("\n=== 4. HEAD vs GET — comparing response sizes ===")

head_response = requests.head(f"{BASE_URL}/posts/1")
get_response  = requests.get(f"{BASE_URL}/posts/1")

print(f"HEAD body length : {len(head_response.content)} bytes")   # 0 bytes
print(f"GET  body length : {len(get_response.content)} bytes")    # actual JSON size
print("HEAD is much faster — no body transferred over the network!")

# ── 5. Practical use: check before downloading a large file ──────────────────
print("\n=== 5. Should I download this file? ===")

url = "https://jsonplaceholder.typicode.com/photos"  # Returns ~500 KB of JSON
response = requests.head(url)

content_length = response.headers.get("Content-Length")

if content_length:
    size_kb = int(content_length) / 1024
    print(f"File size: {size_kb:.1f} KB")
    if size_kb > 100:
        print("Large file — warn user before downloading.")
    else:
        print("Small file — safe to download directly.")
else:
    print("Server did not provide Content-Length — proceed with GET to find out.")

# ── Key things to know ────────────────────────────────────────────────────────
# HEAD response always has an empty body.
# All the headers are identical to what a GET would return.
# Great for: existence checks, size checks, freshness checks.
# Used by browsers to validate cached resources (ETag, Last-Modified).
