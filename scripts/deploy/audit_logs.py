"""
THE HIVE - Full Docker Log Audit
Fetches last N lines from every running container and reports errors/warnings.
"""
import os
import paramiko
import re
import sys
from datetime import datetime

HOST = os.getenv("HIVE_SSH_HOST", "192.168.1.6")
USER = os.getenv("HIVE_SSH_USER", "aza")
PASS = os.getenv("HIVE_SSH_PASSWORD")
SUDO_PASS = os.getenv("HIVE_SUDO_PASSWORD", PASS)
LINES = 80

if not PASS:
    raise RuntimeError("Variable d'environnement HIVE_SSH_PASSWORD manquante.")

ERROR_PATTERNS = re.compile(
    r"(error|exception|traceback|fatal|critical|failed|refused|crash|killed|oom|"
    r"cannot connect|connection refused|timeout|no such|permission denied|"
    r"unauthorized|unhealthy|restart|exit code [^0])",
    re.IGNORECASE,
)
IGNORE_PATTERNS = re.compile(
    r"(level=info|INFO|DEBUG|healthcheck|GET /health|POST /health)",
    re.IGNORECASE,
)


def run(client, cmd, timeout=30):
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    return out, err


def main():
    print(f"\n{'=' * 70}")
    print(f"  THE HIVE - Docker Log Audit | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 70}\n")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS, timeout=15)

    out, _ = run(client, f"echo '{SUDO_PASS}' | sudo -S docker ps --format '{{{{.Names}}}}\t{{{{.Status}}}}\t{{{{.Image}}}}'")
    containers = []
    for line in out.strip().splitlines():
        if "\t" in line:
            parts = line.split("\t")
            containers.append({"name": parts[0], "status": parts[1], "image": parts[2] if len(parts) > 2 else ""})

    print(f"Containers en cours d'execution: {len(containers)}\n")

    all_issues = []

    for ctn in containers:
        name = ctn["name"]
        status = ctn["status"]
        healthy = "OK" if "healthy" in status.lower() else ("WARN" if "unhealthy" in status.lower() else "RUN")

        print(f"\n{'-' * 70}")
        print(f"  [{healthy}] {name}")
        print(f"     Image : {ctn['image']}")
        print(f"     Status: {status}")

        logs, _ = run(client, f"echo '{SUDO_PASS}' | sudo -S docker logs --tail={LINES} --timestamps {name} 2>&1", timeout=20)

        issues = []
        error_lines = []
        for line in logs.splitlines():
            if ERROR_PATTERNS.search(line) and not IGNORE_PATTERNS.search(line):
                error_lines.append(line.strip())
                issues.append({"container": name, "line": line.strip()})

        if error_lines:
            print(f"\n  Anomalies detectees ({len(error_lines)} lignes):")
            for el in error_lines[-10:]:
                print(f"     {el[:120]}")
            all_issues.extend(issues)
        else:
            print(f"  OK Logs propres ({LINES} dernieres lignes analysees)")

    print(f"\n\n{'=' * 70}")
    print("  RESUME DE L'AUDIT")
    print(f"{'=' * 70}")
    print(f"  Containers analyses : {len(containers)}")
    print(f"  Total anomalies     : {len(all_issues)}")

    if all_issues:
        print("\n  Containers avec erreurs:")
        seen = set()
        for issue in all_issues:
            if issue["container"] not in seen:
                count = sum(1 for i in all_issues if i["container"] == issue["container"])
                print(f"     KO {issue['container']} ({count} lignes suspectes)")
                seen.add(issue["container"])
    else:
        print("\n  OK Aucune anomalie critique detectee.")

    print(f"{'=' * 70}\n")
    client.close()


if __name__ == "__main__":
    main()