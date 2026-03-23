import os
import paramiko
import subprocess
import sys

HOST = os.getenv("HIVE_SSH_HOST", "192.168.1.6")
USER = os.getenv("HIVE_SSH_USER", "aza")
PASS = os.getenv("HIVE_SSH_PASSWORD")
SUDO_PASS = os.getenv("HIVE_SUDO_PASSWORD", PASS)

if not PASS:
    raise RuntimeError("Variable d'environnement HIVE_SSH_PASSWORD manquante.")


def deploy():
    print(f"Connecting to Proxmox Server {HOST} to deploy Muse and Nexus...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(HOST, username=USER, password=PASS, timeout=10)
        print("Connected! Fetching remote changes...")

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

        os.remove(tar_path)

        print("Extracting payload on Proxmox...")
        extract_cmd = "cd ~/The_Hive/src/eva-nexus && rm -rf dist && tar -xzf dist.tar.gz && rm dist.tar.gz && echo DONE"
        _, exout, exerr = client.exec_command(extract_cmd)
        print(f"Extraction result: {exout.read().decode()}")
        extract_err = exerr.read().decode()
        if extract_err:
            print(f"Extraction stderr: {extract_err}")

        print("\nStarting Muse and Nexus Docker Containers...")
        docker_cmd = f"cd ~/The_Hive && echo '{SUDO_PASS}' | sudo -S docker compose -f docker-compose.yml up -d --build --no-deps muse nexus"
        stdin, stdout, stderr = client.exec_command(docker_cmd, get_pty=True)

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