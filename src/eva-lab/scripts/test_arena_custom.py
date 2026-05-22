from eva_lab.arena import Arena
import logging

logging.basicConfig(level=logging.INFO)
arena = Arena()
print("Starting arena battle test for scalp horizon...")
report = arena.battle("muzero_scalp_latest", "muzero_scalp_ckpt_1000", horizon="scalp")
print("Battle completed successfully!")
print(f"Outcome: {report.get('outcome')}")
print(f"Challenger Score: {report.get('challenger', {}).get('score')}")
print(f"Champion Score: {report.get('champion', {}).get('score')}")
print(f"Always Long Score: {report.get('always_long_baseline', {}).get('score')}")
print(f"Always Short Score: {report.get('always_short_baseline', {}).get('score')}")
print(f"Hermes Stress Scenario: {report.get('hermes_stress_scenario')}")
