import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("192.168.1.5", username="aza", password="Kumara-42/600", timeout=10)

_, o, _ = client.exec_command("echo 'Kumara-42/600' | sudo -S docker ps --filter name=hive-muse --filter name=hive-nexus --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'")
print(o.read().decode())

client.close()
