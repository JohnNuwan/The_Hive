import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.1.5', username='aza', password='Kumara-42/600')

cmds = [
    "echo '--- df -h ---'",
    "df -h",
    "echo '\n--- lsblk ---'",
    "lsblk -f",
    "echo '\n--- pvesm status ---'",
    "echo 'Kumara-42/600' | sudo -S pvesm status",
    "echo '\n--- zpool status ---'",
    "echo 'Kumara-42/600' | sudo -S zpool status 2>/dev/null || echo 'No zpool'",
    "echo '\n--- Docker Root Dir ---'",
    "docker info -f '{{.DockerRootDir}}'"
]

full_script = " && ".join(cmds)

stdin, stdout, stderr = client.exec_command(full_script)
print(stdout.read().decode())
err = stderr.read().decode()
if err and "password" not in err.lower():
    print(f"ERR: {err}")

client.close()
