"""
Force sync docker-compose.yml on Proxmox and build eva-trainer image.
"""
import paramiko, sys, time

HOST = "192.168.1.5"
USER = "aza"
PASS = "Kumara-42/600"
REMOTE_DIR = "/home/aza/The_Hive"

def run_cmd(client, cmd, timeout=600, print_live=True):
    print(f"\n$ {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd, get_pty=True, timeout=timeout)
    output = ""
    for line in iter(stdout.readline, ""):
        if print_live:
            print(line, end="")
        output += line
    exit_code = stdout.channel.recv_exit_status()
    err = stderr.read().decode()
    if err and exit_code != 0:
        print(f"STDERR: {err[:300]}")
    return exit_code, output

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS, timeout=15)
    print("✅ Connected to Proxmox")

    # 1. Sync everything
    run_cmd(client, f"cd {REMOTE_DIR} && git fetch origin feat/sprint-6 && git reset --hard origin/feat/sprint-6")
    
    # 2. Verify eva-trainer is in docker-compose.yml
    exit_code, out = run_cmd(client, f"grep -n 'eva-trainer' {REMOTE_DIR}/docker-compose.yml")
    if exit_code != 0:
        print("❌ eva-trainer NOT found in docker-compose.yml on server!")
        sys.exit(1)
    print(f"✅ eva-trainer found in compose")
    
    # 3. Set required env vars for docker compose
    env_stub = "MT5_PASSWORD=dummy MT5_LOGIN=123 MT5_SERVER=dummy HUGGING_FACE_HUB_TOKEN=dummy"
    
    print("\n🐳 Building eva-trainer image (Julia + PyTorch + JAX)...")
    print("   This takes ~10-15min the first time.\n")
    # Use sudo (aza is not in docker group yet)
    exit_code, _ = run_cmd(client,
        f"cd {REMOTE_DIR} && {env_stub} echo '{PASS}' | sudo -S docker compose build eva-trainer 2>&1",
        timeout=900)
    
    if exit_code == 0:
        print("\n✅ eva-trainer image built successfully!")
    else:
        print(f"\n❌ Build failed (exit code {exit_code})")
        sys.exit(1)

    # 5. Add aza to docker group (permanent fix, avoids sudo next time)
    run_cmd(client, f"echo '{PASS}' | sudo -S usermod -aG docker aza || true")
    
    # 6. Quick smoke test
    print("\n🧪 Smoke test — checking Python + CUDA access:")
    run_cmd(client, f"cd {REMOTE_DIR} && {env_stub} echo '{PASS}' | sudo -S docker compose run --rm eva-trainer python -c \"import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')\"")
    
    client.close()
    print("\n✅ Done!")

if __name__ == "__main__":
    main()
