import paramiko
import os
import sys

HOST = "192.168.1.5"
USER = "aza"
PASS = "Kumara-42/600"
REMOTE_BASE = "/home/aza/The_Hive"

FILES_TO_UPLOAD = [
    "src/shared/shared/config.py",
    "src/eva-banker/eva_banker/brain.py",
    "src/eva-banker/eva_banker/services/risk.py",
    "src/eva-lab/eva_lab/muzero/config.py",
    "src/eva-lab/scripts/train_global_models.py",
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
    
    # Execute training script in a nohup process or screen so it runs in background
    print("Starting the global training script on the server in the background...")
    # Using nohup to run it detached
    training_cmd = f"cd {REMOTE_BASE} && source venv/bin/activate && nohup python {REMOTE_BASE}/src/eva-lab/scripts/train_global_models.py > {REMOTE_BASE}/train_global.log 2>&1 &"
    
    stdin, stdout, stderr = client.exec_command(training_cmd)
    print("Launched.")
    client.close()
    
if __name__ == "__main__":
    main()
