import paramiko
import os
import sys

HOST = "192.168.1.5"
USER = "aza"
PASS = "Kumara-42/600"
REMOTE_BASE = "/home/aza/The_Hive"

FILES_TO_UPLOAD = [
    "src/openclaw/core/rlm/benchmark.py",
    "src/openclaw/core/rlm/evolver.py",
    "src/eva-builder/eva_builder/main.py", # For the builder's deploy endpoint we analyzed
]

LOCAL_BASE = r"c:\Users\nandi\Desktop\The Hive\The_Hive"

def main():
    print(f"Connecting to {HOST}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, username=USER, password=PASS, timeout=15)
    except Exception as e:
        print(f"Failed to connect: {e}")
        return
        
    sftp = client.open_sftp()
    
    # Upload
    for rel_path in FILES_TO_UPLOAD:
        local_path = os.path.join(LOCAL_BASE, rel_path.replace("/", "\\"))
        remote_path = f"{REMOTE_BASE}/{rel_path}"
        try:
            # Ensure remote dir exists
            remote_dir = os.path.dirname(remote_path)
            stdin, stdout, stderr = client.exec_command(f"mkdir -p {remote_dir}")
            stdout.read()
            
            print(f"Uploading {local_path} -> {remote_path}")
            sftp.put(local_path, remote_path)
        except Exception as e:
            print(f"Error uploading {rel_path}: {e}")
            
    sftp.close()
    
    # Reload Core to pick up the new OpenClaw RLM Evolver and Builder
    print("Restarting Core and Builder containers on the server to apply Sprint 9...")
    docker_cmd = f"cd {REMOTE_BASE} && echo '{PASS}' | sudo -S docker compose restart core builder"
    stdin, stdout, stderr = client.exec_command(docker_cmd)
    
    # Stream the output
    for line in iter(stdout.readline, ""):
        print(line, end="")
        
    print("Sprint 9 successfully deployed!")
    client.close()
    
if __name__ == "__main__":
    main()
