import paramiko
import sys
import os
import subprocess

def deploy():
    print("Connecting to Proxmox Server 192.168.1.5 to deploy Muse and Nexus...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect("192.168.1.5", username="aza", password="Kumara-42/600", timeout=10)
        print("Connected! Fetching remote changes...")
        
        # Git pull
        git_cmds = "cd ~/The_Hive && git fetch origin main && git pull origin main"
        stdin, stdout, stderr = client.exec_command(git_cmds)
        print("Git Output:\n", stdout.read().decode())
        print("Git Errors:\n", stderr.read().decode())
        
        print("Compressing pre-compiled /dist directory...")
        nexus_dir = os.path.join(os.getcwd(), "src", "eva-nexus")
        tar_path = os.path.join(nexus_dir, "dist.tar.gz")
        subprocess.run(["tar", "-czf", "dist.tar.gz", "dist"], cwd=nexus_dir, check=True)
        
        print("Uploading dist.tar.gz to Proxmox via SFTP...")
        sftp = client.open_sftp()
        remote_tar = "/home/aza/The_Hive/src/eva-nexus/dist.tar.gz"
        sftp.put(tar_path, remote_tar)
        sftp.close()
        
        # Clean local tar
        os.remove(tar_path)
        
        print("Extracting payload on Proxmox...")
        extract_cmd = "cd ~/The_Hive/src/eva-nexus && rm -rf dist && tar -xzf dist.tar.gz && rm dist.tar.gz && echo DONE"
        _, exout, exerr = client.exec_command(extract_cmd)
        extract_result = exout.read().decode()
        extract_err = exerr.read().decode()
        print(f"Extraction result: {extract_result}")
        if extract_err:
            print(f"Extraction stderr: {extract_err}")
        
        print("\nStarting Muse and Nexus Docker Containers...")
        # Start docker compose (using --no-deps so it doesn't try to build everything like sentinel)
        docker_cmd = "cd ~/The_Hive && echo 'Kumara-42/600' | sudo -S docker compose -f docker-compose.yml up -d --build --no-deps muse nexus"
        stdin, stdout, stderr = client.exec_command(docker_cmd, get_pty=True)
        
        # Stream output
        for line in iter(stdout.readline, ""):
            print(line, end="")
        
        err = stderr.read().decode()
        if err:
            print(f"Stderr: {err}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        client.close()

if __name__ == "__main__":
    deploy()
