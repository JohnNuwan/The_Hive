import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("192.168.1.5", username="aza", password="Kumara-42/600", timeout=10)

print("Killing docker...")
# pkill matches substring 'docker', so it will kill the docker build process gracefully
stdin, stdout, stderr = client.exec_command("echo 'Kumara-42/600' | sudo -S pkill -f docker")
print(stdout.read().decode())
print(stderr.read().decode())
client.close()
