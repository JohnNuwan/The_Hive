from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path

import paramiko

os.environ['HIVE_SSH_PASSWORD'] = 'Kumara-42/600'
os.environ['HIVE_SUDO_PASSWORD'] = 'Kumara-42/600'

module_path = Path(r'C:\Users\nandi\Desktop\The Hive\The_Hive\scripts\deploy\start_training_proxmox.py')
spec = importlib.util.spec_from_file_location('start_training_proxmox', module_path)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(module)

symbols = module._normalize_scalp_multi_universe_symbols(None)
try:
    source_run_snapshot = module._read_active_remote_run()
except Exception as exc:
    print(f"Lecture HTTP du run source indisponible: {exc}")
    source_run_snapshot = {}
source_run_id = str(source_run_snapshot.get('run_id') or '').strip() or f"manual_{datetime.now():%Y%m%d_%H%M%S}"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(module.HOST, username=module.USER, password=os.environ['HIVE_SSH_PASSWORD'], timeout=15)
try:
    print(f"Attente du vrai checkpoint {module.MANUAL_ARENA_CUTOVER_STEP} pour {source_run_id}...")
    cutover = module._wait_for_remote_checkpoint_cutover(
        client,
        checkpoint_step=module.MANUAL_ARENA_CUTOVER_STEP,
        horizon='scalp',
        timeout_seconds=21600,
        poll_interval_seconds=30,
    )
    print(json.dumps({'cutover': cutover}, indent=2, ensure_ascii=False))

    module.stop_remote_training(client, reason='manual_checkpoint_selection_cutover')
    module._verify_remote_training_stopped(client)

    screen_payload = module._run_remote_manual_screen_arena(
        client,
        source_run_id=source_run_id,
        symbols=symbols,
    )
    print(json.dumps({'screen_payload': screen_payload}, indent=2, ensure_ascii=False, default=float))
    winner = module._select_best_manual_screen_candidate(list(screen_payload.get('screen_results') or []))
    print(json.dumps({'screen_winner': winner}, indent=2, ensure_ascii=False, default=float))

    full_payload = module._run_remote_manual_full_arena(
        client,
        source_run_id=str(screen_payload.get('source_run_id') or source_run_id),
        selected_candidate=winner,
        champion_reference=str(screen_payload.get('champion_reference') or 'gen_000_baseline'),
        symbols=symbols,
    )
    print(json.dumps({'full_payload': full_payload}, indent=2, ensure_ascii=False, default=float))

    if bool(full_payload.get('resume_required')):
        resume_checkpoint_path = str(full_payload.get('resume_checkpoint_path') or '').strip()
        if not resume_checkpoint_path:
            raise RuntimeError('Reprise demandée sans checkpoint.')
        resume_overrides = module._build_manual_resume_after_arena_overrides(
            symbols=symbols,
            resume_checkpoint_path=resume_checkpoint_path,
        )
        try:
            previous_run_id = str((module._read_active_remote_run().get('run_id') or '')).strip() or None
        except Exception:
            previous_run_id = None
        pid = module._launch_remote_training_process(client, resume_overrides)
        print(json.dumps({'resume_launch_pid': pid, 'resume_checkpoint_path': resume_checkpoint_path}, ensure_ascii=False))
        resumed_run_id = module._wait_for_remote_run_start(
            previous_run_id,
            expected_trigger=resume_overrides['TRAINING_RUN_TRIGGER'],
            timeout_seconds=300,
        )
        print(json.dumps({'resumed_run_id': resumed_run_id}, ensure_ascii=False))
finally:
    client.close()
