import paramiko
import sys

def deploy():
    print("Connecting to Proxmox Server 192.168.1.5 to deploy AudioCraft...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect("192.168.1.5", username="aza", password="Kumara-42/600", timeout=10)
        print("Connected! Fetching remote changes...")
        
        # Git pull
        git_cmds = "cd ~/The_Hive && git fetch origin main && git pull origin main"
        stdin, stdout, stderr = client.exec_command(git_cmds)
        print(stdout.read().decode())
        print(stderr.read().decode())
        
        print("Starting AudioCraft Docker Container (this will build the image ~6GB)...")
        # Start docker compose
        docker_cmd = "cd ~/The_Hive && echo 'Kumara-42/600' | sudo -S docker compose -f docker-compose.audiocraft.yml up -d --build"
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
