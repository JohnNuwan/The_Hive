import paramiko
import os

def deploy():
    files = [
        {
            "local": r"C:\Users\nandi\Desktop\The Hive\The_Hive\.env",
            "remote": "/home/aza/The_Hive/.env",
            "container": ["/app/.env"]
        },
        {
            "local": r"C:\Users\nandi\Desktop\The Hive\The_Hive\src\eva-lab\eva_lab\arena.py",
            "remote": "/home/aza/The_Hive/src/eva-lab/eva_lab/arena.py",
            "container": ["/app/eva-lab/eva_lab/arena.py", "/app/eva-lab/build/lib/eva_lab/arena.py"]
        },
        {
            "local": r"C:\Users\nandi\Desktop\The Hive\The_Hive\scripts\hermes_loss_auditor.py",
            "remote": "/home/aza/The_Hive/scripts/hermes_loss_auditor.py",
            "container": ["/app/eva-lab/scripts/hermes_loss_auditor.py"]
        },
        {
            "local": r"C:\Users\nandi\Desktop\The Hive\The_Hive\scripts\apply_alphaevolve_best.py",
            "remote": "/home/aza/The_Hive/scripts/apply_alphaevolve_best.py",
            "container": ["/app/eva-lab/scripts/apply_alphaevolve_best.py"]
        },
        {
            "local": r"C:\Users\nandi\Desktop\The Hive\The_Hive\src\eva-lab\scripts\train_nightly_stack.py",
            "remote": "/home/aza/The_Hive/src/eva-lab/scripts/train_nightly_stack.py",
            "container": ["/app/eva-lab/scripts/train_nightly_stack.py"]
        }
    ]

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        print("Connecting to remote Proxmox server...")
        client.connect('192.168.1.6', username='aza', password='Kumara-42/600', timeout=15)
        
        # Ensure directories exist
        print("Ensuring remote directories exist...")
        dirs_to_make = [
            "/home/aza/The_Hive",
            "/home/aza/The_Hive/src/eva-lab/eva_lab",
            "/home/aza/The_Hive/scripts",
            "/home/aza/The_Hive/src/eva-lab/scripts"
        ]
        for d in dirs_to_make:
            client.exec_command(f"mkdir -p {d}")
            
        sftp = client.open_sftp()
        for f in files:
            print(f"Uploading {os.path.basename(f['local'])} to {f['remote']}...")
            sftp.put(f['local'], f['remote'])
        
        sftp.close()
        print("SFTP Upload completed successfully.")
        
        print("Copying files inside the active container...")
        for f in files:
            for container_path in f["container"]:
                cmd = f"docker cp {f['remote']} the_hive-eva-trainer-run-cdbd0d29e6c3:{container_path}"
                print(f"Running: {cmd}")
                stdin, stdout, stderr = client.exec_command(cmd)
                err = stderr.read().decode('utf-8', errors='replace')
                if err:
                    print(f"Stderr copying to {container_path}: {err}")
                
        print("All copies inside container completed successfully.")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    deploy()
