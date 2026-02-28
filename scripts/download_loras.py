"""
Downloads a curated set of NSFW/Niche LoRAs from CivitAI and uploads them to Proxmox.
Uses the user's CivitAI API key for authenticated downloads.

Models selected to match the defined niches in niches.py.
Add more by finding model IDs at: https://civitai.com/models
NOTE: Only downloads free / authenticated models.
"""
import os
import sys
import paramiko
import tempfile
import httpx
from pathlib import Path

CIVITAI_API_KEY = "bd21426b880b6e020418b6109f312c1d"

# ═══════════════════════════════════════════════════════════════════════════════
# Curated LoRA list — (model_id, version_id, filename, niche)
# These are popular, widely used LoRAs from CivitAI community
# ═══════════════════════════════════════════════════════════════════════════════
LORAS_TO_DOWNLOAD = [
    # Realistic female portrait enhancer (good base for all niches)
    (82098, 87153, "epiNoiseoffset_v2.safetensors", "base"),
    # Add-Detail XL (sharpness enhancer)
    (82098, 87153, "add_detail_xl.safetensors", "base"),
]

# We'll also search CivitAI for popular LoRAs matching our niches using their search API
NICHE_SEARCH_TERMS = [
    ("fitness_athletic.safetensors", "athletic woman fitness photorealistic"),
    ("gfe_girlfriend.safetensors", "girlfriend experience realistic"),
    ("redhead_freckles.safetensors", "red hair freckles natural"),
    ("petite_cute.safetensors", "petite girl cute"),
    ("mature_milf.safetensors", "mature woman elegant realistic"),
]

PROXMOX_HOST = "192.168.1.5"
PROXMOX_USER = "aza"
PROXMOX_PASS = "Kumara-42/600"
REMOTE_LORA_DIR = "/mnt/data/comfyui/models/loras"

def search_civitai(query: str, limit: int = 1) -> list[dict]:
    """Search CivitAI for LoRA models matching a query."""
    url = "https://civitai.com/api/v1/models"
    params = {
        "limit": limit,
        "query": query,
        "types": "LORA",
        "sort": "Most Downloaded",
        "nsfw": "true",
    }
    headers = {"Authorization": f"Bearer {CIVITAI_API_KEY}"}
    with httpx.Client(timeout=30.0) as client:
        res = client.get(url, params=params, headers=headers)
        data = res.json()
        return data.get("items", [])


def download_lora(version_id: int) -> bytes:
    """Downloads a LoRA from CivitAI by model version ID."""
    url = f"https://civitai.com/api/download/models/{version_id}"
    headers = {"Authorization": f"Bearer {CIVITAI_API_KEY}"}
    with httpx.Client(timeout=300.0, follow_redirects=True) as client:
        res = client.get(url, headers=headers)
        res.raise_for_status()
        return res.content


def upload_to_proxmox(ssh: paramiko.SSHClient, data: bytes, filename: str):
    """Uploads bytes to Proxmox LoRA directory."""
    remote_path = f"{REMOTE_LORA_DIR}/{filename}"
    sftp = ssh.open_sftp()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        size_mb = len(data) / 1024 / 1024
        print(f"  → Uploading {filename} ({size_mb:.1f} MB)...")
        sftp.put(tmp_path, remote_path)
        print(f"  ✅ {filename}")
    finally:
        sftp.close()
        os.unlink(tmp_path)


def main():
    print(f"🔑 CivitAI API Key: {CIVITAI_API_KEY[:8]}...")
    print(f"🔗 Connecting to Proxmox {PROXMOX_HOST}...")
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(PROXMOX_HOST, username=PROXMOX_USER, password=PROXMOX_PASS)
    ssh.exec_command(f"mkdir -p {REMOTE_LORA_DIR}")
    print("✅ Connected.\n")

    # Search and download niche LoRAs via CivitAI search API
    print("🔍 Searching CivitAI for niche-matched LoRAs...")
    for filename, query in NICHE_SEARCH_TERMS:
        print(f"\n  Searching: '{query}'")
        try:
            results = search_civitai(query, limit=1)
            if not results:
                print(f"  ⚠️  No results for '{query}'")
                continue
            
            model = results[0]
            version = model.get("modelVersions", [{}])[0]
            version_id = version.get("id")
            model_name = model.get("name", "Unknown")
            
            if not version_id:
                print(f"  ⚠️  No version found for '{model_name}'")
                continue
            
            print(f"  Found: {model_name} (v{version_id})")
            data = download_lora(version_id)
            upload_to_proxmox(ssh, data, filename)
                
        except Exception as e:
            print(f"  ❌ Failed: {e}")

    # Verify what's now in the LoRA dir
    _, o, _ = ssh.exec_command(f"ls -lh {REMOTE_LORA_DIR}")
    print(f"\n📂 Current LoRAs on Proxmox ({REMOTE_LORA_DIR}):")
    print(o.read().decode())

    ssh.close()
    print("✅ CivitAI LoRA download complete.")
    print("▶ Update infra/comfyui/loras_catalog.json with new filenames to activate them.")


if __name__ == "__main__":
    main()
