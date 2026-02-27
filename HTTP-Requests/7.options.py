import requests

# =============================================================================
# HTTP OPTIONS REQUEST
# =============================================================================
# PURPOSE : Ask the server "what HTTP methods do you support for this URL?"
#           Also used by browsers as a CORS "preflight" check before sending
#           a cross-origin request.
# SAFE    : Yes — never modifies anything.
# IDEMPOTENT: Yes — same result every time.
# BODY    : No.
#
# REAL-WORLD USE CASES:
#   1. CORS Preflight — browser auto-sends OPTIONS before cross-origin POST/PUT
#   2. API Discovery   — find out what methods an endpoint accepts
#   3. Debugging       — confirm server config / CORS headers are correct
#
# HOW CORS PREFLIGHT WORKS:
#   Browser → OPTIONS /api/data (with Origin + Access-Control-Request-Method)
#   Server  → 200 OK with Access-Control-Allow-* headers
#   Browser → (if allowed) sends the actual POST/PUT request
# =============================================================================

# ── 1. Basic OPTIONS — what methods are allowed? ──────────────────────────────
print("=== 1. Discover allowed methods (OPTIONS /posts) ===")

response = requests.options("https://jsonplaceholder.typicode.com/posts")

print("Status Code    :", response.status_code)  # 200 or 204
print("Allow header   :", response.headers.get("Allow", "Not provided"))
# The 'Allow' header lists all HTTP methods the server accepts for this URL
# e.g. "GET, POST, PUT, PATCH, DELETE, OPTIONS"

print("All headers:")
for key, value in response.headers.items():
    print(f"  {key}: {value}")

# ── 2. Simulating a CORS preflight request ────────────────────────────────────
print("\n=== 2. CORS Preflight — what a browser sends automatically ===")

# When your React/Vue app (on localhost:3000) calls an API (on api.example.com),
# the browser first sends this OPTIONS request to ask for permission.

cors_headers = {
    "Origin"                        : "http://localhost:3000",      # Where the request comes from
    "Access-Control-Request-Method" : "POST",                       # The method you want to use
    "Access-Control-Request-Headers": "Content-Type, Authorization" # Headers you want to send
}

response = requests.options(
    "https://jsonplaceholder.typicode.com/posts",
    headers=cors_headers
)

print("Status Code                    :", response.status_code)
print("Access-Control-Allow-Origin    :", response.headers.get("Access-Control-Allow-Origin", "Not set"))
print("Access-Control-Allow-Methods   :", response.headers.get("Access-Control-Allow-Methods", "Not set"))
print("Access-Control-Allow-Headers   :", response.headers.get("Access-Control-Allow-Headers", "Not set"))
print("Access-Control-Max-Age         :", response.headers.get("Access-Control-Max-Age", "Not set"))
# Max-Age tells browser how many seconds to cache this preflight result
# (avoids sending OPTIONS before every single request)

# ── 3. Check if a real CORS-enabled API has proper headers ───────────────────
print("\n=== 3. Verify CORS is configured correctly on an API ===")

response = requests.options("https://httpbin.org/anything", headers={
    "Origin": "http://myapp.com",
    "Access-Control-Request-Method": "DELETE"
})

allow_origin = response.headers.get("Access-Control-Allow-Origin")
allow_methods = response.headers.get("Access-Control-Allow-Methods")

print("Allow-Origin  :", allow_origin)
print("Allow-Methods :", allow_methods)

if allow_origin in ("*", "http://myapp.com"):
    print("CORS is configured — cross-origin requests are allowed!")
else:
    print("CORS not configured for this origin — browser would block the request.")

# ── 4. What happens if CORS fails ────────────────────────────────────────────
print("\n=== 4. When CORS fails (educational) ===")
print("""
Timeline of a CORS failure:
  1. Browser sends: OPTIONS /api/data
     Headers: Origin: http://localhost:3000
              Access-Control-Request-Method: POST

  2. Server responds WITHOUT Access-Control-Allow-Origin header
     (or with wrong origin)

  3. Browser BLOCKS the actual POST from being sent.
     Console error: "CORS policy: No 'Access-Control-Allow-Origin' header"

Fix: Server must respond to OPTIONS with:
     Access-Control-Allow-Origin: http://localhost:3000   (or *)
     Access-Control-Allow-Methods: GET, POST, PUT, DELETE
     Access-Control-Allow-Headers: Content-Type, Authorization
""")

# ── Key things to know ────────────────────────────────────────────────────────
# OPTIONS is automatic — browsers send it before cross-origin requests.
# You rarely write OPTIONS calls yourself; they happen behind the scenes.
# Look for 'Allow' header to know what methods a URL supports.
# CORS errors are always OPTIONS-related — check those headers when debugging.
