import paramiko
import sys

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.1.5', username='aza', password='Kumara-42/600')

cmds = [
    # Stop Docker and Containerd
    "echo 'Kumara-42/600' | sudo -S systemctl stop docker docker.socket containerd",
    
    # Move the huge containerd folder to the 1TB drive (fast mv since it's same machine, but crossing filesystems takes time, use mv or cp/rm)
    # Using mv will automatically copy and remove. This might take a few minutes for 384GB.
    "echo 'Kumara-42/600' | sudo -S mv /var/lib/containerd /mnt/data/containerd_payload",
    "echo 'Kumara-42/600' | sudo -S mv /var/lib/docker /mnt/data/docker_payload",
    
    # Create symlinks
    "echo 'Kumara-42/600' | sudo -S ln -s /mnt/data/containerd_payload /var/lib/containerd",
    "echo 'Kumara-42/600' | sudo -S ln -s /mnt/data/docker_payload /var/lib/docker",
    
    # Restart services
    "echo 'Kumara-42/600' | sudo -S systemctl start containerd docker",
    
    # Check free space on root 
    "df -h /"
]

for cmd in cmds:
    print(f"=== {cmd} ===")
    stdin, stdout, stderr = client.exec_command(cmd)
    
    # Block and wait for command to finish (necessary for large mv)
    exit_status = stdout.channel.recv_exit_status()
    
    print(stdout.read().decode())
    err = stderr.read().decode()
    if err and "password" not in err.lower():
        print(f"ERR: {err}")

client.close()
