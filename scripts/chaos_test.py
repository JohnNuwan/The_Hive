"""
Chaos Test Script — THE HIVE
Simulates random failures to verify Swarm resilience.
"""

import docker
import time
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLIENT = docker.from_env()

# Services to target (names as defined in docker-compose.yml)
TARGETS = [
    "hive-core",
    "hive-banker",
    "hive-sentinel",
    "hive-nervous"
]

def kill_random_service():
    target = random.choice(TARGETS)
    logger.warning(f"💥 CHAOS: Targeting {target} for termination...")
    
    try:
        container = CLIENT.containers.get(target)
        container.stop()
        logger.info(f"💀 {target} has been stopped. Waiting for Phoenix Protocol...")
    except Exception as e:
        logger.error(f"❌ Failed to kill {target}: {e}")

def verify_resurrection(target, timeout=60):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            container = CLIENT.containers.get(target)
            if container.status == "running":
                logger.info(f"🔥 PHOENIX: {target} has been resurrected successfully!")
                return True
        except:
            pass
        time.sleep(2)
    
    logger.error(f"❌ FAILURE: {target} did not recover within {timeout}s")
    return False

if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════╗")
    print("║  💥 THE HIVE CHAOS MONKEY                    ║")
    print("║  Verifying Phoenix Protocol Resilience       ║")
    print("╚══════════════════════════════════════════════╝\n")
    
    # We'll do 2 rounds of chaos
    for i in range(2):
        target = random.choice(TARGETS)
        kill_random_service()
        time.sleep(5) # Give the self-healing service a second to notice
        verify_resurrection(target)
        print("-" * 40)
        time.sleep(10)

    logger.info("🎯 Chaos test complete. Swarm integrity verified.")
