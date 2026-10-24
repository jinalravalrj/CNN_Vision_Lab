"""
tests/test_api.py
─────────────────
Basic smoke tests for the CNN Vision Lab API.
Requires the Flask server to be running on http://127.0.0.1:5000.

Usage:
    python tests/test_api.py
"""

import io
import json
import sys
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:5000"

PASS = 0
FAIL = 0


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  ✅  {label}")
        PASS += 1
    else:
        print(f"  ❌  {label}" + (f" — {detail}" if detail else ""))
        FAIL += 1


def get(path):
    req = urllib.request.urlopen(BASE + path, timeout=10)
    return json.loads(req.read())


def post_multipart(path, field, filename, content, content_type="image/jpeg"):
    boundary = "----TestBoundary1234"
    body  = f'--{boundary}\r\n'.encode()
    body += f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'.encode()
    body += f'Content-Type: {content_type}\r\n\r\n'.encode()
    body += content
    body += f'\r\n--{boundary}--\r\n'.encode()

    req = urllib.request.Request(
        BASE + path,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        res = urllib.request.urlopen(req, timeout=60)
        return json.loads(res.read()), res.getcode()
    except urllib.error.HTTPError as e:
        return json.loads(e.read()), e.code


def make_tiny_png():
    """Return a minimal valid 1×1 white PNG as bytes."""
    import struct, zlib
    def chunk(name, data):
        c = name + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    idat = chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
    iend = chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


def main():
    print("=" * 55)
    print("  CNN Vision Lab — API Smoke Tests")
    print("=" * 55)

    # ── Health ──────────────────────────────────────────────
    print("\n[1] Health endpoint")
    try:
        data = get("/api/health")
        check("Status is 'ok'", data.get("status") == "ok")
        check("model_loaded field present", "model_loaded" in data)
        if not data.get("model_loaded"):
            print("      ⚠️  Model not loaded yet — prediction tests may fail.")
    except Exception as e:
        check("Health endpoint reachable", False, str(e))

    # ── Model info ──────────────────────────────────────────
    print("\n[2] Model info endpoint")
    try:
        data = get("/api/model-info")
        check("model_name present", bool(data.get("model_name")))
        check("output_classes == 1000", data.get("output_classes") == 1000)
    except Exception as e:
        check("Model info reachable", False, str(e))

    # ── History (empty or existing) ─────────────────────────
    print("\n[3] History endpoint")
    try:
        data = get("/api/history")
        check("success == True", data.get("success") is True)
        check("records is a list", isinstance(data.get("records"), list))
    except Exception as e:
        check("History endpoint reachable", False, str(e))

    # ── Predict — valid PNG ─────────────────────────────────
    print("\n[4] Predict — valid 1×1 PNG")
    try:
        png = make_tiny_png()
        data, code = post_multipart("/api/predict", "image", "test.png", png, "image/png")
        check("HTTP 200", code == 200, f"got {code}")
        check("success == True", data.get("success") is True, str(data.get("error")))
        if data.get("success"):
            check("predicted field present", bool(data.get("predicted")))
            check("confidence in [0, 100]", 0 <= data.get("confidence", -1) <= 100)
            check("top3 has 3 items", len(data.get("top3", [])) == 3)
            check("infer_ms > 0", data.get("infer_ms", 0) > 0)
    except Exception as e:
        check("Predict endpoint reachable", False, str(e))

    # ── Predict — missing file ──────────────────────────────
    print("\n[5] Predict — no file (should return 400)")
    try:
        req = urllib.request.Request(
            BASE + "/api/predict",
            data=b"",
            headers={"Content-Type": "multipart/form-data; boundary=x"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=10)
            check("Returns 400 for missing file", False, "Expected 400 but got 200")
        except urllib.error.HTTPError as e:
            check("Returns 4xx for missing file", e.code in (400, 500), f"got {e.code}")
    except Exception as e:
        check("Missing-file test", False, str(e))

    # ── Predict — invalid extension ─────────────────────────
    print("\n[6] Predict — .txt file (should reject)")
    try:
        data, code = post_multipart("/api/predict", "image", "bad.txt", b"hello", "text/plain")
        check("Rejects unsupported type", not data.get("success"))
    except Exception as e:
        check("Invalid-extension test", False, str(e))

    # ── Summary ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print("=" * 55)
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
