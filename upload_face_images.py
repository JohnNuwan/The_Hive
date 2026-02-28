"""
Moves face images from the temporary path to the correct ComfyUI Docker volume path,
and also uploads them fresh from local if needed.
"""
import paramiko
import os

ARTIFACTS_DIR = r"C:\Users\nandi\.gemini\antigravity\brain\31aa86b8-74bc-4cc1-9dc7-f7f184eb5f54"

FACE_IMAGES = {
    "neo_face.jpg": "neo_face_1772281987802.png",
    "lois_face.jpg": "lois_face_1772282006822.png",
    "athena_face.jpg": "athena_face_1772282019342.png",
}

# The Docker compose mounts /mnt/data/comfyui/input -> /opt/ComfyUI/input inside container
CORRECT_INPUT = "/mnt/data/comfyui/input"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("192.168.1.5", username="aza", password="Kumara-42/600", timeout=10)

# Ensure the directory exists (the deploy script should have created it)
_, o, e = client.exec_command(f"echo 'Kumara-42/600' | sudo -S mkdir -p {CORRECT_INPUT} && echo 'Kumara-42/600' | sudo -S chmod -R 777 {CORRECT_INPUT} && echo OK")
print(o.read().decode())

print(f"Uploading face images directly to {CORRECT_INPUT}...")
sftp = client.open_sftp()
for remote_name, local_filename in FACE_IMAGES.items():
    local_path = os.path.join(ARTIFACTS_DIR, local_filename)
    remote_path = f"{CORRECT_INPUT}/{remote_name}"
    print(f"  Uploading {local_filename} -> {remote_path}")
    sftp.put(local_path, remote_path)
    print(f"  ✅ Done")

sftp.close()

# Verify
_, o, _ = client.exec_command(f"ls -lh {CORRECT_INPUT}")
print("\nContent of ComfyUI input/:")
print(o.read().decode())

client.close()
print("✅ Face images are in the correct ComfyUI volume path!")
