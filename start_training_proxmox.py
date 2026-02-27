import paramiko
import sys

def start_training():
    print("Connecting to Proxmox Server 192.168.1.5 to start training...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect("192.168.1.5", username="aza", password="Kumara-42/600", timeout=10)
        print("Connected! Launching offline_trainer.py in the background inside the Lab container...")
        
        # We use docker compose exec with -d (detached) to run the training in the background
        train_cmd = "cd ~/The_Hive && echo 'Kumara-42/600' | sudo -S docker compose exec -e PYTHONPATH=/app/eva-lab -d lab python eva_lab/muzero/offline_trainer.py"
        stdin, stdout, stderr = client.exec_command(train_cmd)
        
        output = stdout.read().decode()
        error = stderr.read().decode()
        
        if output:
            print("Output:", output)
        if error and not error.startswith("[sudo] password for"):
            # Sudo prompts usually go to stderr, but if it's something else we show it
            print("Stderr:", error)
            
        print("✅ Training launched successfully on the server!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        client.close()

if __name__ == "__main__":
    start_training()
