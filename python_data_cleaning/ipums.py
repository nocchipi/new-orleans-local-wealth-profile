from __future__ import annotations

import json
import os
import time
import pathlib
from typing import Any, Dict

import requests


API_KEY = os.getenv("IPUMS_API_KEY", "").strip()
print(API_KEY)
if not API_KEY:
    raise SystemExit("Set IPUMS_API_KEY in your environment.")

COLLECTION = "usa"
VERSION = "beta"  # as shown in IPUMS v1 workflow examples

BASE = "https://api.ipums.org"
OUT_DIR = pathlib.Path("./ipums_downloads").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)


def headers() -> Dict[str, str]:
    # IMPORTANT: v1 uses Authorization: <API_KEY> (no "Bearer")
    return {
        "Authorization": API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }



def submit_extract(payload: Dict[str, Any]) -> int:
    url = f"{BASE}/extracts"
    params = {"collection": COLLECTION, "version": VERSION}

    h = headers()

    # Hard fail if the header isn't present
    if not isinstance(h, dict) or "Authorization" not in h or not h["Authorization"]:
        raise RuntimeError(f"headers() did not return a usable Authorization header. Got: {h}")

    # Build & inspect the prepared request (shows EXACT outgoing headers)
    req = requests.Request("POST", url, params=params, headers=h, json=payload)
    prepped = req.prepare()

    print("Outgoing request URL:", prepped.url)
    print("Outgoing Authorization header present?:", "Authorization" in prepped.headers)
    if "Authorization" in prepped.headers:
        v = prepped.headers["Authorization"]
        print("Outgoing Authorization len:", len(v))
        print("Outgoing Authorization preview:", v[:6] + "..." + v[-4:])

    with requests.Session() as s:
        r = s.send(prepped, timeout=60, allow_redirects=False)

    print("Status:", r.status_code)
    print("Body:", r.text)

    # If IPUMS responds with redirect, follow manually (preserving headers)
    if r.status_code in (301, 302, 307, 308) and "Location" in r.headers:
        loc = r.headers["Location"]
        print("Redirect to:", loc)
        r = requests.post(loc, headers=h, json=payload, timeout=60)
        print("Redirected status:", r.status_code)
        print("Redirected body:", r.text)

    if r.status_code == 403:
        raise PermissionError(
            "403 Forbidden from IPUMS. Common causes:\n"
            " - Your IPUMS account is not registered/approved for this collection\n"
            " - Wrong API key\n"
            f"Response body: {r.text}"
        )

    r.raise_for_status()
    data = r.json()

    if "number" not in data:
        raise RuntimeError(f"Unexpected response (no 'number'): {data}")
    return int(data["number"])


def get_extract(extract_number: int) -> Dict[str, Any]:
    url = f"{BASE}/extracts/{extract_number}"
    params = {"collection": COLLECTION, "version": VERSION}
    r = requests.get(url, params=params, headers=headers(), timeout=60)
    r.raise_for_status()
    return r.json()


def wait_for_extract(extract_number: int, poll_seconds: int = 15, timeout_seconds: int = 60 * 60) -> Dict[str, Any]:
    start = time.time()
    while True:
        info = get_extract(extract_number)
        status = (info.get("status") or "").lower()
        print(f"[extract {extract_number}] status={status}")

        # v1 statuses include: queued, started, produced, canceled, failed, completed :contentReference[oaicite:4]{index=4}
        if status == "completed":
            return info
        if status in {"failed", "canceled"}:
            raise RuntimeError(f"Extract {extract_number} ended with status={status}: {info}")

        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for extract {extract_number}")

        time.sleep(poll_seconds)


def download_file(url: str, dest_path: pathlib.Path) -> pathlib.Path:
    # download_links URLs are direct file URLs; include Authorization header when downloading as well :contentReference[oaicite:5]{index=5}
    with requests.get(url, headers={"Authorization": API_KEY}, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return dest_path


def main() -> None:
    # IMPORTANT: v1 expects "samples" and "variables" as objects (dicts), not lists :contentReference[oaicite:6]{index=6}
    extract_payload: Dict[str, Any] = {
        "description": "My IPUMS USA API-submitted extract",
        "data_structure": {"rectangular": {"on": "P"}},  # common for person-level rectangular extracts :contentReference[oaicite:7]{index=7}
        "data_format": "fixed_width",
        "samples": {
            "us2022a": {}   # change to your sample(s)
        },
        "variables": {
            "AGE": {},
            "SEX": {},
            "RACE": {},
            "HISPAN": {},
            "STATEFIP": {},
        },
    }

    print("Submitting extract...")
    num = submit_extract(extract_payload)
    print("Extract number:", num)

    print("Waiting for completion...")
    final = wait_for_extract(num)

    # When complete, download_links contains the file URLs (ddi, syntax, data, etc.) :contentReference[oaicite:8]{index=8}
    links = final.get("download_links", {})
    if not links:
        raise RuntimeError(f"No download_links found on completed extract: {final}")

    print("Downloading files:")
    for name, meta in links.items():
        file_url = meta["url"]
        filename = pathlib.Path(file_url).name
        dest = OUT_DIR / filename
        print(" -", name, "->", dest.name)
        download_file(file_url, dest)

    print("Done. Files saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
