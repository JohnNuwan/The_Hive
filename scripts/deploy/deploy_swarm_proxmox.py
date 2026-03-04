import paramiko
import sys
import time

HOST = "192.168.1.5"
USER = "aza"
PASS = "Kumara-42/600"
REMOTE_DIR = "/home/aza/The_Hive"
STACK_NAME = "hive"

def deploy_swarm():
    print(f"Connecting to Proxmox Swarm Manager {HOST}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(HOST, username=USER, password=PASS, timeout=15)
        print("Connected! Fetching remote changes...")
        
        # Pull latest code
        git_cmds = f"cd {REMOTE_DIR} && git fetch origin feat/sprint-6 && git reset --hard origin/feat/sprint-6"
        stdin, stdout, stderr = client.exec_command(git_cmds)
        print(stdout.read().decode())
        
        print("Ensuring Swarm Mode is active...")
        swarm_init_cmd = f"echo '{PASS}' | sudo -S docker swarm init || true"
        client.exec_command(swarm_init_cmd)
        time.sleep(2)
        
        print(f"Deploying Stack '{STACK_NAME}' via docker-compose.yml...")
        # In Swarm mode, we must pre-build images or use a registry. Since we build locally on the node:
        # We first build via compose, then deploy via stack.
        
        build_cmd = f"cd {REMOTE_DIR} && echo '{PASS}' | sudo -S docker compose build"
        print("Building images (this may take a while)...")
        stdin, stdout, stderr = client.exec_command(build_cmd, get_pty=True)
        for line in iter(stdout.readline, ""):
            print(line, end="")
            
        # Deploy with standard Docker Compose (required for NVIDIA GPU devices allocation)
        deploy_cmd = f"cd {REMOTE_DIR} && echo '{PASS}' | sudo -S docker compose up -d --remove-orphans"
        stdin, stdout, stderr = client.exec_command(deploy_cmd, get_pty=True)
        for line in iter(stdout.readline, ""):
            print(line, end="")
            
        # Check status
        status_cmd = f"cd {REMOTE_DIR} && echo '{PASS}' | sudo -S docker compose ps"
        stdin, stdout, stderr = client.exec_command(status_cmd)
        print("\nDocker Compose Service Status:")
        print(stdout.read().decode())
        
    except Exception as e:
        print(f"Deployment failed: {e}")
        sys.exit(1)
    finally:
        client.close()
        print("Connection closed.")

if __name__ == "__main__":
    deploy_swarm()
