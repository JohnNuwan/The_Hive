"""
Manual step: installs VHS and ReActor using sudo git (to avoid auth issues) on Proxmox.
Run this if install_comfyui_nodes.py failed with auth issues.
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
    ("ComfyUI-VideoHelperSuite", "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"),
    ("comfyui-reactor-node", "https://github.com/Gourieff/comfyui-reactor-node"),
]

for name, url in nodes:
    cmd = (
        f"echo '{SUDO_PASS}' | sudo -S bash -c 'if [ ! -d {CUSTOM_NODES_DIR}/{name} ]; then "
        f"git clone {url} {CUSTOM_NODES_DIR}/{name} && echo DONE; else echo ALREADY_EXISTS; fi'"
    )
    _, o, e = client.exec_command(cmd)
    print(f"{name}:", o.read().decode(), e.read().decode())

_, o, _ = client.exec_command(f"echo '{SUDO_PASS}' | sudo -S docker restart eva_comfyui && echo RESTARTED")
print("ComfyUI:", o.read().decode())
client.close()