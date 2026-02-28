import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.1.5', username='aza', password='Kumara-42/600')

cmd = "lsblk -o NAME,FSTYPE,SIZE,MOUNTPOINT,LABEL"
print("=== lsblk ===")
stdin, stdout, stderr = client.exec_command(cmd)
print(stdout.read().decode().strip())

client.close()
