# HTTP Requests — Quick Reference

## What is HTTP?
A protocol for communication between a **client** (browser/app) and a **server**. Every request has a **method**, **URL**, **headers**, and optionally a **body**.

---

## HTTP Methods

| Method | Purpose | Has Body | Idempotent | Safe |
|--------|---------|----------|------------|------|
| GET | Read/fetch data | No | Yes | Yes |
| POST | Create new resource | Yes | No | No |
| PUT | Replace entire resource | Yes | Yes | No |
| PATCH | Partially update resource | Yes | No | No |
| DELETE | Delete a resource | No | Yes | No |
| HEAD | Like GET but no body returned | No | Yes | Yes |
| OPTIONS | Get allowed methods for a URL | No | Yes | Yes |

---

## Anatomy of an HTTP Request

```
GET /users/1 HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Content-Type: application/json
```

**Parts:**
- **Method** — what action
- **URL/Path** — where
- **Headers** — metadata (auth, content type, etc.)
- **Body** — data payload (POST/PUT/PATCH only)

---

## HTTP Status Codes

| Range | Meaning | Common Codes |
|-------|---------|--------------|
| 1xx | Informational | 100 Continue |
| 2xx | Success | 200 OK, 201 Created, 204 No Content |
| 3xx | Redirect | 301 Moved Permanently, 304 Not Modified |
| 4xx | Client Error | 400 Bad Request, 401 Unauthorized, 403 Forbidden, 404 Not Found, 429 Too Many Requests |
| 5xx | Server Error | 500 Internal Server Error, 502 Bad Gateway, 503 Service Unavailable |

---

## Key Headers

| Header | Usage |
|--------|-------|
| `Content-Type` | Format of request body (`application/json`) |
| `Accept` | Format client expects in response |
| `Authorization` | Auth token (`Bearer <token>`, `Basic ...`) |
| `Cache-Control` | Caching directives |
| `CORS` headers | Cross-origin access control |

---

## Key Concepts for Interviews

**Idempotent** — Calling it N times = same result as calling it once. (GET, PUT, DELETE are idempotent. POST is not.)

**Safe** — Does not modify server state. (GET, HEAD, OPTIONS are safe.)

**REST** — Architectural style using HTTP methods + nouns as URLs.

**Stateless** — Every HTTP request is independent. No memory of previous requests (use tokens/cookies to persist identity).

**CORS** — Browser security policy. Server must allow cross-origin requests via headers like `Access-Control-Allow-Origin`.

**HTTP vs HTTPS** — HTTPS encrypts traffic using TLS. Always use HTTPS in production.

**HTTP/1.1 vs HTTP/2** — HTTP/2 supports multiplexing (multiple requests over one connection), header compression, and server push.

---

## Common Interview Questions

1. **Difference between PUT and PATCH?**  
   PUT replaces the entire resource. PATCH updates only specified fields.

2. **Why is POST not idempotent?**  
   Each POST creates a new resource — calling it twice creates two records.

3. **401 vs 403?**  
   401 = Not authenticated (no/invalid token). 403 = Authenticated but not authorized.

4. **What happens in a GET request?**  
   Client sends request → DNS resolves URL → TCP connection → TLS handshake (HTTPS) → Server returns response.

5. **What is a preflight request?**  
   Browser sends OPTIONS request before CORS request to check if server allows it.