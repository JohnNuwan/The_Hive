import paramiko
import sys

def deploy():
    print("Connecting to Proxmox Server 192.168.1.5 to deploy ComfyUI...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect("192.168.1.5", username="aza", password="Kumara-42/600", timeout=10)
        print("Connected! Fetching remote changes...")
        
        # Git pull
        git_cmds = "cd ~/The_Hive && git fetch origin main && git reset --hard origin/main"
        stdin, stdout, stderr = client.exec_command(git_cmds)
        print(stdout.read().decode())
        print(stderr.read().decode())
        
        print("Creating ComfyUI directories on /mnt/data...")
        dir_cmds = "echo 'Kumara-42/600' | sudo -S mkdir -p /mnt/data/comfyui/{output,input,models,custom_nodes} && echo 'Kumara-42/600' | sudo -S chmod -R 777 /mnt/data/comfyui"
        client.exec_command(dir_cmds)
        
        print("Installing ReActor FaceSwap Custom Node...")
        reactor_cmd = "if [ ! -d /mnt/data/comfyui/custom_nodes/comfyui-reactor-node ]; then git clone https://github.com/Gourieff/comfyui-reactor-node /mnt/data/comfyui/custom_nodes/comfyui-reactor-node; fi"
        client.exec_command(reactor_cmd)
        
        print("Starting ComfyUI Docker Container with NVIDIA GPU access...")
        # Start docker compose
        docker_cmd = "cd ~/The_Hive && echo 'Kumara-42/600' | sudo -S docker compose -f docker-compose.comfyui.yml up -d"
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
