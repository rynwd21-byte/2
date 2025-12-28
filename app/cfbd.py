import os, httpx, json

BASE = "https://api.collegefootballdata.com"
API_KEY = os.getenv("CFBD_API_KEY")


def _headers():
    h = {"Accept": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


async def get(path, params=None):
    """Call the CollegeFootballData API and return parsed JSON.

    This wrapper is intentionally defensive:
    - If CFBD_API_KEY is missing, we raise a clear RuntimeError.
    - If CFBD returns a non-2xx status, we raise a RuntimeError with the
      status code and a short excerpt of the response body.
    - If the body is not valid JSON (for example, HTML swagger docs),
      we raise a RuntimeError instead of letting a JSONDecodeError bubble
      out of the ASGI stack.
    """
    if not API_KEY:
        raise RuntimeError("CFBD_API_KEY is not set in environment.")

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{BASE}{path}", params=params or {}, headers=_headers())

        # HTTP-level errors first
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:  # type: ignore[attr-defined]
            snippet = r.text[:300]
            raise RuntimeError(
                f"CFBD error {r.status_code} at {path}: {snippet}"
            ) from exc

        # Now try to parse JSON; if this fails, surface a readable error
        try:
            return r.json()
        except json.JSONDecodeError as exc:
            snippet = r.text[:300]
            raise RuntimeError(
                f"CFBD returned non-JSON at {path}: {snippet}"
            ) from exc
