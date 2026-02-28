import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.1.5', username='aza', password='Kumara-42/600')

cmds = [
    "echo 'Kumara-42/600' | sudo -S du -cd 1 /var 2>/dev/null | sort -hr | head -n 10",
    "echo '---'",
    "echo 'Kumara-42/600' | sudo -S du -cd 1 /home 2>/dev/null | sort -hr | head -n 10",
]

for cmd in cmds:
    print(f"=== {cmd} ===")
    stdin, stdout, stderr = client.exec_command(cmd)
    
    print(stdout.read().decode())
    err = stderr.read().decode()
    if err and "password" not in err.lower():
        print(f"ERR: {err}")

client.close()
