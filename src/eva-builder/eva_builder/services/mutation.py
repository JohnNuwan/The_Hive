"""
Service Mutation - THE HIVE
Triggers the evolution pipeline after code modifications.
"""

import asyncio
import logging
import subprocess
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class MutationService:
    def __init__(self, root_dir: str = "/app/the_hive"):
        self.root_dir = Path(root_dir)
        self.runner_path = self.root_dir / "scripts" / "evolution_runner.py"

    async def trigger_evolution(self, change_summary: str) -> dict:
        """
        Triggers the evolution runner script.
        """
        logger.info(f"🧬 MutationService: Triggering evolution for: {change_summary}")
        
        if not self.runner_path.exists():
            error = f"Evolution runner not found at {self.runner_path}"
            logger.error(error)
            return {"status": "error", "message": error}

        try:
            # Run the evolution runner in a sub-process
            process = await asyncio.create_subprocess_exec(
                "python", str(self.runner_path),
                cwd=str(self.root_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info("✅ Evolution successful.")
                return {
                    "status": "success",
                    "output": stdout.decode(),
                    "summary": change_summary
                }
            else:
                logger.error(f"❌ Evolution failed: {stderr.decode()}")
                return {
                    "status": "failed",
                    "error": stderr.decode(),
                    "output": stdout.decode()
                }

        except Exception as e:
            logger.error(f"Mutation failure: {e}")
            return {"status": "error", "message": str(e)}
