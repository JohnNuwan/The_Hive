"""
Quick network diagnostic — checks Redis reachability from within a container
and checks vLLM GPU status.
"""
import paramiko

HOST = "192.168.1.5"
USER = "aza"
PASS = "Kumara-42/600"

def run(client, cmd, timeout=30):
    _, s, e = client.exec_command(cmd, timeout=timeout)
    return (s.read() + e.read()).decode(errors='replace').strip()

def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASS, timeout=15)
    print("✅ Connected\n")

    # 1. Redis container status
    print("=== Redis Status ===")
    out = run(c, f"echo '{PASS}' | sudo -S docker inspect --format='ID={{{{.ID[:12]}}}} Status={{{{.State.Status}}}} IP={{{{range .NetworkSettings.Networks}}}}{{{{.IPAddress}}}}{{{{end}}}}' the_hive_lite-redis-1 2>/dev/null || echo 'Not found'")
    print(out)

    # 2. Check actual Redis port binding
    print("\n=== Redis Port Binding ===")
    out = run(c, f"echo '{PASS}' | sudo -S docker port the_hive_lite-redis-1 2>/dev/null || echo 'N/A'")
    print(out)

    # 3. Check if redis responds from the host
    print("\n=== Redis Ping from host ===")
    out = run(c, f"echo '{PASS}' | sudo -S docker exec the_hive_lite-redis-1 redis-cli -a devpassword ping 2>/dev/null || echo 'FAILED'")
    print(f"Redis PING response: {out}")

    # 4. Check overlay network
    print("\n=== Hive Network Containers ===")
    out = run(c, f"echo '{PASS}' | sudo -S docker network inspect the_hive_lite_hive-net --format='{{{{range .Containers}}}}{{{{.Name}}}} → {{{{.IPv4Address}}}}\\n{{{{end}}}}' 2>/dev/null | head -30")
    print(out if out else "Network not found or empty")

    # 5. vLLM GPU check
    print("\n=== vLLM Container GPU ===")
    out = run(c, f"echo '{PASS}' | sudo -S docker inspect --format='Status={{{{.State.Status}}}} RestartCount={{{{.RestartCount}}}}' the_hive_lite-vllm-1 2>/dev/null")
    print(out)
    out = run(c, f"echo '{PASS}' | sudo -S docker logs --tail=20 the_hive_lite-vllm-1 2>&1 | grep -i 'cuda\\|gpu\\|error\\|failed' | head -10")
    print(out)

    # 6. Mosquitto (MQTT) status
    print("\n=== Mosquitto/MQTT ===")
    out = run(c, f"echo '{PASS}' | sudo -S docker inspect --format='Status={{{{.State.Status}}}} IP={{{{range .NetworkSettings.Networks}}}}{{{{.IPAddress}}}}{{{{end}}}}' the_hive_lite-mosquitto-1 2>/dev/null || echo 'N/A'")
    print(out)

    # 7. Loki (for promtail)
    print("\n=== Loki ===")
    out = run(c, f"echo '{PASS}' | sudo -S docker ps --filter name=loki --format '{{{{.Names}}}} {{{{.Status}}}}' 2>/dev/null")
    print(out if out else "Loki not running")

    c.close()
    print("\n✅ Diagnostic terminé.")

if __name__ == "__main__":
    main()
