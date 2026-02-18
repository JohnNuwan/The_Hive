import sys
import io
import contextlib
import logging
import traceback
import ast
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Whitelist of safe modules
SAFE_MODULES = {
    "math",
    "random",
    "datetime",
    "json",
    "re",
    "collections",
    "itertools",
    "functools",
    "typing",
}

# Dangerous attributes to block via AST check
DANGEROUS_ATTRS = {
    "__class__",
    "__bases__",
    "__subclasses__",
    "__globals__",
    "__code__",
    "__closure__",
    "__func__",
    "__self__",
    "__module__",
    "__dict__",
    "__builtins__",
    "func_globals",
    "func_code",
    "func_closure",
    "__import__",
    "format",
    "format_map",
}

class SecurityScanner(ast.NodeVisitor):
    def visit_Attribute(self, node):
        if node.attr in DANGEROUS_ATTRS:
            raise SecurityError(f"Access to attribute '{node.attr}' is forbidden.")
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id == "__builtins__":
             raise SecurityError("Access to '__builtins__' is forbidden.")
        self.generic_visit(node)

class SecurityError(Exception):
    pass

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name in SAFE_MODULES:
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Import of module '{name}' is restricted by CyberForge security policy.")

class SafeLogger:
    """Safe wrapper for logging to preventing access to underlying Logger internals."""
    def info(self, msg):
        logger.info(msg)

    def error(self, msg):
        logger.error(msg)

    def warning(self, msg):
        logger.warning(msg)

    def debug(self, msg):
        logger.debug(msg)

# Whitelist of safe builtins
# Removed: object, type, getattr, setattr, delattr, property, staticmethod, classmethod, super
# to prevent introspection and attribute manipulation bypasses.
SAFE_BUILTINS = {
    "__import__": guarded_import,
    "abs": abs,
    "all": all,
    "any": any,
    "ascii": ascii,
    "bin": bin,
    "bool": bool,
    "bytearray": bytearray,
    "bytes": bytes,
    "callable": callable,
    "chr": chr,
    "complex": complex,
    "dict": dict,
    "dir": dir,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "frozenset": frozenset,
    "hasattr": hasattr,
    "hash": hash,
    "help": help,
    "hex": hex,
    "id": id,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "vars": vars, # vars() without args is locals(), with args is __dict__. Might be risky if allowed on objects.
    "zip": zip,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "AttributeError": AttributeError,
    "NameError": NameError,
    "SyntaxError": SyntaxError,
    "RuntimeError": RuntimeError,
    "ImportError": ImportError,
}

# vars(obj) is equivalent to obj.__dict__. Since we block __dict__, we should block vars on objects.
# vars() returns locals.
# To be safe, let's remove vars from builtins.
if "vars" in SAFE_BUILTINS:
    del SAFE_BUILTINS["vars"]

class CyberForge:
    """
    La Forge de l'Architecte.
    Permet à E.V.A. de coder et tester ses propres scripts d'analyse.
    """
    def __init__(self):
        self.history = []

    def _validate_context(self, context: Dict) -> None:
        """Ensures context only contains safe data types."""
        ALLOWED_TYPES = (str, int, float, bool, list, dict, tuple, set, type(None))
        for key, value in context.items():
            if not isinstance(value, ALLOWED_TYPES):
                raise SecurityError(f"Context variable '{key}' has unsafe type '{type(value).__name__}'.")

    def forge_and_test(self, script_name: str, code: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Exécute un script généré dans un environnement supervisé et restreint.
        """
        logger.info(f"CyberForge: Forging script '{script_name}'...")
        
        # Capture de la sortie standard
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        success = False
        output = ""
        error = None

        try:
            # 1. AST Static Analysis
            tree = ast.parse(code)
            scanner = SecurityScanner()
            scanner.visit(tree)

            # 2. Prepare restricted execution environment
            exec_globals = {"__builtins__": SAFE_BUILTINS.copy()}

            if context:
                self._validate_context(context)
                exec_globals.update(context)

            # Provide safe logger wrapper
            exec_globals.update({
                "__name__": "__cyberforge__",
                "logger": SafeLogger(),
            })

            # 3. Execution supervisée
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                exec(code, exec_globals)
            
            success = True
            output = stdout_capture.getvalue()
            logger.info(f"CyberForge: Script '{script_name}' executed successfully.")
        except Exception:
            success = False
            # Capture both validation errors and execution errors
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
