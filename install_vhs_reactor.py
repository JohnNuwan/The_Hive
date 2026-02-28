"""
Manual step: installs VHS and ReActor using sudo git (to avoid auth issues) on Proxmox.
Run this if install_comfyui_nodes.py failed with 'could not read Username'.
"""
import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("192.168.1.5", username="aza", password="Kumara-42/600", timeout=10)

CUSTOM_NODES_DIR = "/mnt/data/comfyui/custom_nodes"

nodes = [
    ("ComfyUI-VideoHelperSuite", "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"),
    ("comfyui-reactor-node", "https://github.com/Gourieff/comfyui-reactor-node"),
]

for name, url in nodes:
    cmd = f"echo 'Kumara-42/600' | sudo -S bash -c 'if [ ! -d {CUSTOM_NODES_DIR}/{name} ]; then git clone {url} {CUSTOM_NODES_DIR}/{name} && echo DONE; else echo ALREADY_EXISTS; fi'"
    _, o, e = client.exec_command(cmd)
    print(f"{name}:", o.read().decode(), e.read().decode())

# Restart ComfyUI
_, o, _ = client.exec_command("echo 'Kumara-42/600' | sudo -S docker restart eva_comfyui && echo RESTARTED")
print("ComfyUI:", o.read().decode())
client.close()
