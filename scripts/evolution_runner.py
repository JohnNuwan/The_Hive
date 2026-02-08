"""
THE HIVE — Auto-Evolution Runner (CI/CD)
This script orchestrates the build, test, and deployment of swarm mutations.
"""

import subprocess
import sys
import os
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EvolutionRunner:
    def __init__(self):
        self.repo_path = os.getcwd()
        self.test_scripts = [
            "verify_logic.py",
            "scripts/chaos_test.py"
        ]

    def run_command(self, cmd, shell=True):
        logger.info(f"🚀 Executing: {cmd}")
        result = subprocess.run(cmd, shell=shell, text=True, capture_output=True)
        if result.returncode != 0:
            logger.error(f"❌ Command failed: {result.stderr}")
            return False, result.stderr
        return True, result.stdout

    def verify_mutation(self):
        """Run all verification scripts."""
        logger.info("🧪 Starting Verification Suite...")
        
        # 1. Syntax & Logic Check
        success, output = self.run_command("python scripts/verify_logic.py")
        if not success: return False
        
        # 2. Swarm Health Check
        success, output = self.run_command("python scripts/verify_health.py")
        if not success: return False
        
        # 3. Chaos Resilience Check
        success, output = self.run_command("python scripts/chaos_test.py")
        if not success: return False
        
        return True

    def deploy_mutation(self, commit_msg="Mutation applied by Auto-Evolution"):
        """Commit and push to hive_auto branch, then restart services."""
        logger.info("📦 Deploying Mutation to 'hive_auto' branch...")
        
        # Git Cycle
        self.run_command("git add .")
        
        # Ensure we are on hive_auto branch
        success, current_branch = self.run_command("git rev-parse --abbrev-ref HEAD")
        target_branch = "hive_auto"
        
        # Create or switch to hive_auto
        self.run_command(f"git checkout -b {target_branch} || git checkout {target_branch}")
        
        # Commit
        self.run_command(f'git commit -m "{commit_msg}"')
        
        # Push to remote (assuming origin exists)
        self.run_command(f"git push origin {target_branch}")
        
        # Return to original branch (if it wasn't hive_auto)
        if current_branch.strip() != target_branch:
             self.run_command(f"git checkout {current_branch.strip()}")

        # Swarm Cycle
        self.run_command("docker-compose up -d --build")
        
        logger.info(f"✨ Evolution Complete. Mutation pushed to {target_branch} and swarm redeployed.")

    def run_pipeline(self):
        logger.info("🛠️ Evolution Pipeline Triggered")
        
        if self.verify_mutation():
            logger.info("✅ Verification Passed")
            self.deploy_mutation()
        else:
            logger.error("❌ Evolution Failed: Verification did not pass. Reverting...")
            self.run_command("git checkout .")

if __name__ == "__main__":
    runner = EvolutionRunner()
    runner.run_pipeline()
