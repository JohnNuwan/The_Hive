import paramiko
import sys

def debug_remote():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect("192.168.1.5", username="aza", password="Kumara-42/600", timeout=10)
        
        print("--- Docker PS ---")
        stdin, stdout, stderr = client.exec_command("echo 'Kumara-42/600' | sudo -S docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'")
        print(stdout.read().decode())
        
        print("--- Testing API locally on server ---")
        stdin, stdout, stderr = client.exec_command("curl -s http://localhost:8600/openapi.json | grep -o '\"/gnn/predict\"'")
        print("GNN Predict Route in OpenAPI: ", stdout.read().decode().strip())
        
        stdin, stdout, stderr = client.exec_command("curl -s http://localhost:8600/docs")
        print("Docs length: ", len(stdout.read().decode()))

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        client.close()

if __name__ == "__main__":
    debug_remote()
