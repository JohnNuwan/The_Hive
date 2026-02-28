import paramiko
import sys

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.1.5', username='aza', password='Kumara-42/600')

cmds = [
    # Verify sdb has no filesystem
    "echo 'Kumara-42/600' | sudo -S wipefs -a /dev/sdb",
    # Format to ext4
    "echo 'Kumara-42/600' | sudo -S mkfs.ext4 -F /dev/sdb",
    # Create mount point
    "echo 'Kumara-42/600' | sudo -S mkdir -p /mnt/data",
    # Mount it
    "echo 'Kumara-42/600' | sudo -S mount /dev/sdb /mnt/data",
    # Add to fstab if not present
    "echo 'Kumara-42/600' | sudo -S su -c 'grep -q \"/dev/sdb /mnt/data\" /etc/fstab || echo \"/dev/sdb /mnt/data ext4 defaults 0 0\" >> /etc/fstab'",
    "df -h /mnt/data"
]

for cmd in cmds:
    print(f"=== {cmd} ===")
    stdin, stdout, stderr = client.exec_command(cmd)
    
    # Wait for the command to finish if it's mkfs
    exit_status = stdout.channel.recv_exit_status()
    print(stdout.read().decode())
    err = stderr.read().decode()
    if err and "password" not in err.lower():
        print(f"ERR: {err}")

client.close()
