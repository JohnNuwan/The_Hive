@echo off
setlocal
set "HIVE_SSH_PASSWORD=Kumara-42/600"
set "HIVE_SUDO_PASSWORD=Kumara-42/600"
python scripts\deploy\run_scalp_multi_universe_chain.py
