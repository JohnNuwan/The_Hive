import sys
import io
import logging
import traceback
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class CyberForge:
    """
    La Forge de l'Architecte.
    Permet à E.V.A. de coder et tester ses propres scripts d'analyse.
    """
    def __init__(self):
        self.history = []

    def forge_and_test(self, script_name: str, code: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Exécute un script généré dans un environnement supervisé.
        """
        logger.info(f"CyberForge: Forging script '{script_name}'...")
        
        # Capture de la sortie standard
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        exec_globals = context or {}
        exec_globals.update({
            "__name__": "__cyberforge__",
            "logger": logger,
        })

        success = False
        output = ""
        error = None

        try:
            # Exécution supervisée
            with io.redirect_stdout(stdout_capture), io.redirect_stderr(stderr_capture):
                exec(code, exec_globals)
            
            success = True
            output = stdout_capture.getvalue()
            logger.info(f"CyberForge: Script '{script_name}' executed successfully.")
        except Exception:
            success = False
            error = traceback.format_exc()
            logger.error(f"CyberForge Error in '{script_name}': {error}")
        finally:
            stdout_capture.close()
            stderr_capture.close()

        result = {
            "script_name": script_name,
            "success": success,
            "output": output,
            "error": error
        }
        
        self.history.append(result)
        return result

    def get_forge_history(self):
        return self.history
