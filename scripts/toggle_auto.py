import urllib.request
import os
import json
import time
import jwt # Requires PyJWT

URL = "http://localhost:8100/trading/auto"
SECRET = os.environ.get("INTERNAL_SECRET_KEY")

def generate_token():
    if not SECRET:
        raise ValueError("INTERNAL_SECRET_KEY environment variable is not set")
    payload = {
        "iss": "hive-test-script",
        "sub": "internal-swarm-request",
        "src": "script",
        "iat": int(time.time()),
        "exp": int(time.time()) + 60
    }
    return jwt.encode(payload, SECRET, algorithm="HS256")

def toggle():
    token = generate_token()
    headers = {
        'Content-Type': 'application/json',
        'X-Hive-Internal-Token': token
    }

    # 1. Disable
    try:
        print("Disabling Auto-Trading...")
        req = urllib.request.Request(URL, data=json.dumps({"enable": False}).encode('utf-8'), headers=headers)
        with urllib.request.urlopen(req) as f:
            print(f.read().decode('utf-8'))
    except Exception as e:
        print(f"Error disabling: {e}")

    time.sleep(2)

    # 2. Enable
    try:
        print("Enabling Auto-Trading...")
        req = urllib.request.Request(URL, data=json.dumps({"enable": True}).encode('utf-8'), headers=headers)
        with urllib.request.urlopen(req) as f:
            print(f.read().decode('utf-8'))
    except Exception as e:
        print(f"Error enabling: {e}")

if __name__ == "__main__":
    toggle()
