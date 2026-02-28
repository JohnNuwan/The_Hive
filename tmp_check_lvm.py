import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.1.5', username='aza', password='Kumara-42/600')

cmds = [
    "echo '--- lsblk ---'",
    "lsblk -f",
    "echo '--- pvs ---'",
    "echo 'Kumara-42/600' | sudo -S pvs",
    "echo '--- vgs ---'",
    "echo 'Kumara-42/600' | sudo -S vgs",
    "echo '--- lvs ---'",
    "echo 'Kumara-42/600' | sudo -S lvs"
]

full_script = " && ".join(cmds)

stdin, stdout, stderr = client.exec_command(full_script)
print(stdout.read().decode())
err = stderr.read().decode()
if err and "password" not in err.lower():
    print(f"ERR: {err}")

client.close()
