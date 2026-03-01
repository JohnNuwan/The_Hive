import paramiko
import sys

def debug_remote():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect("192.168.1.5", username="aza", password="Kumara-42/600", timeout=10)
        
        print("--- Docker PS ---")
        stdin, stdout, stderr = client.exec_command("echo 'Kumara-42/600' | sudo -S docker ps -a | grep lab")
        print(stdout.read().decode())
        
        print("--- Docker Logs (tail 50) ---")
        stdin, stdout, stderr = client.exec_command("echo 'Kumara-42/600' | sudo -S docker logs --tail 50 hive-lab")
        print(stdout.read().decode())
        print("ERRORS:", stderr.read().decode())
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        client.close()

if __name__ == "__main__":
    debug_remote()
