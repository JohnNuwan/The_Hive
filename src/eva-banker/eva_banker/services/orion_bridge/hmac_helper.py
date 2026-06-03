import hashlib
import hmac
import json
import os
import time


def generate_hmac_signature(payload: str) -> tuple[str, str]:
    """
    Generate Hmac signature from the payload + timestamp and secret key
    Params -> payload: dict
    Return -> str: timestamp, str: signature

    """

    timestamp = str(int(time.time()))

    secret_key = os.getenv("BRIDGE_SECRET_KEY")
    if not secret_key:
        raise ValueError("BRIDGE_SECRET_KEY environment variable is not set")

    bytes_secret_key = secret_key.encode("utf-8")
    message = f"{timestamp}.{payload}".encode("utf-8")
    signature = hmac.new(bytes_secret_key, message, hashlib.sha256).hexdigest()

    return timestamp, signature
