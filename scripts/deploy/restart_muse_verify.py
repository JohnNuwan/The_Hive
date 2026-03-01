"""Diagnose ComfyUI port availability from the muse container perspective."""
import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.1.5', username='aza', password='Kumara-42/600')

def run(cmd):
    stdin, stdout, stderr = c.exec_command(cmd)
    stdin.write("Kumara-42/600\n")
    stdin.flush()
    return stdout.read().decode('utf-8', errors='replace').strip()

# 1. Check ComfyUI container status
print("=== ComfyUI containers ===")
print(run("sudo -S docker ps --format '{{.Names}}\\t{{.Ports}}\\t{{.Status}}' | grep -i comfy"))

# 2. Check if port 8188 is listening on the host
print("\n=== Listening on 8188 ===")
print(run("ss -tlnp | grep 8188"))

# 3. Try curl from inside muse container to ComfyUI
print("\n=== Curl ComfyUI from muse container ===")
print(run("sudo -S docker exec hive-muse curl -s http://192.168.1.5:8188/system_stats --max-time 3 2>&1 | head -c 200"))

# 4. Try curl from host directly
print("\n=== Curl ComfyUI from host ===")
print(run("curl -s http://localhost:8188/system_stats --max-time 3 2>&1 | head -c 200"))

# 5. Check docker networks
print("\n=== Docker networks ComfyUI ===")
print(run("sudo -S docker inspect eva_comfyui --format '{{json .HostConfig.PortBindings}}' 2>/dev/null || echo 'Container eva_comfyui not found'"))

c.close()
