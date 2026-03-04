"""
THE HIVE — Docker Full Log Audit (writes report to /tmp/hive_audit.txt on server)
"""
import paramiko, re

HOST = "192.168.1.5"
USER = "aza"
PASS = "Kumara-42/600"

TARGETS = [
    "the_hive_lite-compliance-1",
    "the_hive_lite-substrate-1",
    "the_hive_lite-shadow-2",
    "the_hive_lite-rwa-2",
    "the_hive_lite-kernel-2",
    "the_hive_lite-promtail-1",
    "the_hive_lite-comfyui-1",
    "the_hive_lite-vllm-1",
    "the_hive_lite-builder-1",
    "the_hive_lite-accountant-2",
]

ERR = re.compile(r'(error|exception|traceback|fatal|critical|failed|no module|refused|crash|killed|oom|cannot connect|timeout|unhealthy|exit code [^0])', re.I)
OK  = re.compile(r'(level=info|"level":"info"|DEBUG|GET /health|POST /health|ping)', re.I)

def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASS, timeout=15)

    report_lines = ["=" * 72, "  THE HIVE — Docker Audit Report", "=" * 72]
    summary = []

    for name in TARGETS:
        report_lines += [f"\n{'─'*72}", f"  📦 {name}", f"{'─'*72}"]

        _, s, _ = c.exec_command(f"echo '{PASS}' | sudo -S docker inspect --format='Status={{{{.State.Status}}}} Restarts={{{{.RestartCount}}}} ExitCode={{{{.State.ExitCode}}}}' {name} 2>/dev/null")
        status = s.read().decode().strip()
        report_lines.append(f"  {status}")

        _, s, e = c.exec_command(f"echo '{PASS}' | sudo -S docker logs --tail=150 {name} 2>&1", timeout=30)
        logs = s.read().decode(errors='replace') + e.read().decode(errors='replace')

        err_lines = [l.strip() for l in logs.splitlines() if ERR.search(l) and not OK.search(l)]
        last10 = logs.strip().splitlines()[-10:]

        if err_lines:
            report_lines.append(f"\n  🚨 {len(err_lines)} lignes suspectes (dernières 15):")
            for l in err_lines[-15:]:
                report_lines.append(f"     {l[:130]}")
            summary.append(f"  ❌ {name}: {len(err_lines)} erreurs")
        else:
            report_lines.append("  ✅ Aucune erreur détectée")
            summary.append(f"  ✅ {name}: OK")

        report_lines.append(f"\n  📋 Dernières lignes:")
        for l in last10:
            report_lines.append(f"     {l[:130]}")

    report_lines += ["\n" + "=" * 72, "  RÉSUMÉ", "=" * 72] + summary + ["=" * 72]

    report_text = "\n".join(report_lines)

    # Write to remote temp file
    sftp = c.open_sftp()
    with sftp.open("/tmp/hive_audit.txt", "w") as f:
        f.write(report_text)
    sftp.close()

    # Also print locally
    print(report_text)
    c.close()

if __name__ == "__main__":
    main()
