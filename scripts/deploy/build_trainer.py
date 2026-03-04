"""
Script to create required directories and verify the eva-trainer Docker image.
Also tests a direct docker run (bypassing Swarm).
"""
import paramiko, sys

HOST = "192.168.1.5"
USER = "aza"
PASS = "Kumara-42/600"
REMOTE_DIR = "/home/aza/The_Hive"

def run_cmd(client, cmd, timeout=300, print_live=True):
    print(f"\n$ {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd, get_pty=True, timeout=timeout)
    output = ""
    for line in iter(stdout.readline, ""):
        if print_live:
            print(line, end="")
        output += line
    exit_code = stdout.channel.recv_exit_status()
    return exit_code, output

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS, timeout=15)
    print("✅ Connected to Proxmox")

    # 1. Pull latest
    run_cmd(client, f"cd {REMOTE_DIR} && git pull origin feat/sprint-6")
    
    # 2. Create required data directory on host for model weights
    run_cmd(client, f"mkdir -p {REMOTE_DIR}/src/eva-lab/data/models")
    print("✅ Data directory created")

    # 3. Create Swarm volume storage path if needed
    run_cmd(client, f"echo '{PASS}' | sudo -S mkdir -p /mnt/data/docker_payload/volumes && echo OK || true")
    
    # 4. Build image (should be cached — fast)
    run_cmd(client, f"cd {REMOTE_DIR} && echo '{PASS}' | sudo -S docker compose build eva-trainer", timeout=600)
    
    # 5. Smoke test using plain docker run (bypasses Swarm compose layer)
    print("\n🧪 Testing with plain docker run (no Swarm):")
    exit_code, _ = run_cmd(client,
        f"echo '{PASS}' | sudo -S docker run --rm --gpus all "
        f"-v {REMOTE_DIR}/src/eva-lab:/app/eva-lab "
        f"thehive/eva-trainer:latest "
        f"python -c \"import torch; "
        f"print('✅ CUDA:', torch.cuda.is_available(), "
        f"'GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')\"",
        timeout=120
    )
    if exit_code == 0:
        print("\n✅ eva-trainer works with direct docker run!")
    else:
        print(f"\n⚠️ Exit code {exit_code}")
    
    client.close()
    print("\n✅ Done!")

if __name__ == "__main__":
    main()
