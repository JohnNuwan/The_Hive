"""
CivitAI LoRA Downloader — Downloads LoRAs to Proxmox's ComfyUI models directory.

Usage:
  CIVITAI_API_KEY=your_key python download_loras.py

Add LoRA model IDs to LORAS_TO_DOWNLOAD below.
Find model IDs on https://civitai.com/models/<ID>
"""
import os
import sys
import paramiko
import tempfile
import httpx

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — Add CivitAI model IDs here
# Format: (civitai_model_id, civitai_version_id, target_filename)
# Find version_id in the URL: civitai.com/models/<model_id>?modelVersionId=<version_id>
# ═══════════════════════════════════════════════════════════════════════════════
LORAS_TO_DOWNLOAD = [
    # (model_id, version_id, filename)
    # Examples — replace with real IDs from civitai.com:
    # (123456, 789012, "fitness_athletic_v2.safetensors"),
    # (234567, 891234, "bdsm_dominant_lora.safetensors"),
    # Add your IDs here:
]

CIVITAI_API_KEY = os.getenv("CIVITAI_API_KEY", "")
PROXMOX_HOST = "192.168.1.5"
PROXMOX_USER = "aza"
PROXMOX_PASS = "Kumara-42/600"
REMOTE_LORA_DIR = "/mnt/data/comfyui/models/loras"


def download_lora_from_civitai(model_id: int, version_id: int) -> bytes:
    """Downloads a LoRA from CivitAI and returns its bytes."""
    url = f"https://civitai.com/api/download/models/{version_id}"
    headers = {"Authorization": f"Bearer {CIVITAI_API_KEY}"} if CIVITAI_API_KEY else {}
    
    print(f"Downloading from CivitAI model {model_id} version {version_id}...")
    with httpx.Client(timeout=300.0, follow_redirects=True) as client:
        res = client.get(url, headers=headers)
        res.raise_for_status()
        return res.content


def upload_to_proxmox(client: paramiko.SSHClient, data: bytes, remote_filename: str):
    """Uploads bytes to the Proxmox LoRA directory via SFTP."""
    remote_path = f"{REMOTE_LORA_DIR}/{remote_filename}"
    sftp = client.open_sftp()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    
    try:
        print(f"Uploading {remote_filename} to Proxmox ({len(data) // 1024 // 1024}MB)...")
        sftp.put(tmp_path, remote_path)
        print(f"✅ {remote_filename} uploaded to {remote_path}")
    finally:
        sftp.close()
        os.unlink(tmp_path)


def main():
    if not LORAS_TO_DOWNLOAD:
        print("⚠️  No LoRAs configured in LORAS_TO_DOWNLOAD.")
        print("   Edit download_loras.py and add CivitAI model IDs.")
        print("   Find IDs at: https://civitai.com/models")
        sys.exit(0)

    if not CIVITAI_API_KEY:
        print("⚠️  No CIVITAI_API_KEY set. Some models may not download without authentication.")
        print("   Set it with: $env:CIVITAI_API_KEY='your_key_here'")

    # Connect to Proxmox
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(PROXMOX_HOST, username=PROXMOX_USER, password=PROXMOX_PASS)

    # Ensure target dir exists
    ssh.exec_command(f"mkdir -p {REMOTE_LORA_DIR}")

    for model_id, version_id, filename in LORAS_TO_DOWNLOAD:
        try:
            data = download_lora_from_civitai(model_id, version_id)
            upload_to_proxmox(ssh, data, filename)
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")

    ssh.close()
    print("\n✅ All downloads complete.")
    print("Update infra/comfyui/loras_catalog.json to enable the new LoRAs.")


if __name__ == "__main__":
    main()
