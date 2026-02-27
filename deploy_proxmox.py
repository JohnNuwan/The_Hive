import paramiko
import time
import sys

def deploy():
    print("Connecting to Proxmox Server 192.168.1.5...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect("192.168.1.5", username="aza", password="Kumara-42/600", timeout=10)
        print("Connected! Fetching remote changes and resetting branch...")
        
        # Git commands
        git_cmds = "cd ~/The_Hive && git fetch origin main && git reset --hard origin/main"
        stdin, stdout, stderr = client.exec_command(git_cmds)
        print(stdout.read().decode())
        print(stderr.read().decode())
        
        print("Rebuilding Docker containers Lab and Accountant...")
        # Docker build commands
        docker_cmd = "cd ~/The_Hive && echo 'Kumara-42/600' | sudo -S docker compose -f docker-compose.yml up -d --build lab accountant"
        stdin, stdout, stderr = client.exec_command(docker_cmd, get_pty=True)
        
        # Stream the output for better visibility
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
