"""
Installs required ComfyUI custom nodes for video generation on Proxmox:
- AnimateDiff-Evolved (video motion)
- ComfyUI-VideoHelperSuite / VHS (video combining/saving)
"""
import os
import paramiko

HOST = os.getenv("HIVE_SSH_HOST", "192.168.1.6")
USER = os.getenv("HIVE_SSH_USER", "aza")
PASS = os.getenv("HIVE_SSH_PASSWORD")
SUDO_PASS = os.getenv("HIVE_SUDO_PASSWORD", PASS)

if not PASS:
    raise RuntimeError("Variable d'environnement HIVE_SSH_PASSWORD manquante.")

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)

CUSTOM_NODES_DIR = "/mnt/data/comfyui/custom_nodes"

nodes = [
    ("AnimateDiff-Evolved", "https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git"),
    ("ComfyUI-VideoHelperSuite", "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"),
    ("comfyui-reactor-node", "https://github.com/Gourieff/comfyui-reactor-node.git"),
]

for name, url in nodes:
    cmd = f"""
if [ ! -d {CUSTOM_NODES_DIR}/{name} ]; then
    echo "Installing {name}..."
    git clone {url} {CUSTOM_NODES_DIR}/{name}
    echo "Installing Python requirements..."
    pip install -r {CUSTOM_NODES_DIR}/{name}/requirements.txt 2>/dev/null || true
    echo "Done: {name}"
else
    echo "{name} already installed, pulling updates..."
    cd {CUSTOM_NODES_DIR}/{name} && git pull
fi
"""
    _, o, e = client.exec_command(cmd)
    print(o.read().decode())
    err = e.read().decode()
    if err:
        print(f"Stderr: {err}")

print("Restarting ComfyUI container...")
_, o, _ = client.exec_command(f"echo '{SUDO_PASS}' | sudo -S docker restart eva_comfyui && echo OK")
print(o.read().decode())

client.close()
print("ComfyUI custom nodes installed. AnimateDiff and VHS are ready.")