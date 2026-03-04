"""
THE HIVE — Full Docker Log Audit
Fetches last N lines from every running container and reports errors/warnings.
"""
import paramiko, sys, re
from datetime import datetime

HOST = "192.168.1.5"
USER = "aza"
PASS = "Kumara-42/600"
LINES = 80  # Lines per container

ERROR_PATTERNS = re.compile(
    r'(error|exception|traceback|fatal|critical|failed|refused|crash|killed|oom|'
    r'cannot connect|connection refused|timeout|no such|permission denied|'
    r'unauthorized|unhealthy|restart|exit code [^0])',
    re.IGNORECASE
)
IGNORE_PATTERNS = re.compile(
    r'(level=info|INFO|DEBUG|healthcheck|GET /health|POST /health)',
    re.IGNORECASE
)

def run(client, cmd, timeout=30):
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors='replace')
    err = stderr.read().decode(errors='replace')
    return out, err

def main():
    print(f"\n{'='*70}")
    print(f"  🐝 THE HIVE — Docker Log Audit | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}\n")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS, timeout=15)
    
    # Get container list
    out, _ = run(client, f"echo '{PASS}' | sudo -S docker ps --format '{{{{.Names}}}}\t{{{{.Status}}}}\t{{{{.Image}}}}'")
    containers = []
    for line in out.strip().splitlines():
        if '\t' in line:
            parts = line.split('\t')
            containers.append({'name': parts[0], 'status': parts[1], 'image': parts[2] if len(parts) > 2 else ''})
    
    print(f"📦 {len(containers)} containers en cours d'exécution:\n")
    
    all_issues = []
    
    for c in containers:
        name = c['name']
        status = c['status']
        healthy = '✅' if 'healthy' in status.lower() else ('⚠️ ' if 'unhealthy' in status.lower() else '🔄')
        
        print(f"\n{'─'*70}")
        print(f"  {healthy} {name}")
        print(f"     Image : {c['image']}")
        print(f"     Status: {status}")
        
        # Fetch logs
        logs, _ = run(client, f"echo '{PASS}' | sudo -S docker logs --tail={LINES} --timestamps {name} 2>&1", timeout=20)
        
        issues = []
        error_lines = []
        for line in logs.splitlines():
            if ERROR_PATTERNS.search(line) and not IGNORE_PATTERNS.search(line):
                error_lines.append(line.strip())
                issues.append({'container': name, 'line': line.strip()})
        
        if error_lines:
            print(f"\n  🚨 ANOMALIES DÉTECTÉES ({len(error_lines)} lignes):")
            for el in error_lines[-10:]:  # Show last 10 error lines
                print(f"     {el[:120]}")
            all_issues.extend(issues)
        else:
            print(f"  ✅ Logs propres ({LINES} dernières lignes analysées)")
    
    # Summary
    print(f"\n\n{'='*70}")
    print(f"  📊 RÉSUMÉ DE L'AUDIT")
    print(f"{'='*70}")
    print(f"  Containers analysés : {len(containers)}")
    print(f"  Total anomalies     : {len(all_issues)}")
    
    if all_issues:
        print(f"\n  🚨 Containers avec erreurs:")
        seen = set()
        for issue in all_issues:
            if issue['container'] not in seen:
                count = sum(1 for i in all_issues if i['container'] == issue['container'])
                print(f"     ❌ {issue['container']} ({count} lignes suspectes)")
                seen.add(issue['container'])
    else:
        print(f"\n  ✅ Aucune anomalie critique détectée. La Ruche est saine !")
    
    print(f"{'='*70}\n")
    client.close()

if __name__ == "__main__":
    main()
