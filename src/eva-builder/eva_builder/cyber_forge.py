"""Execution securisee de scripts Python pour `eva-builder`."""

from __future__ import annotations

import ast
import contextlib
import io
import logging
import traceback
from typing import Any

logger = logging.getLogger(__name__)

SAFE_MODULES = {
    "collections",
    "datetime",
    "functools",
    "itertools",
    "json",
    "math",
    "random",
    "re",
    "typing",
}

DANGEROUS_ATTRS = {
    "__bases__",
    "__builtins__",
    "__class__",
    "__closure__",
    "__code__",
    "__dict__",
    "__func__",
    "__globals__",
    "__import__",
    "__module__",
    "__self__",
    "__subclasses__",
    "func_closure",
    "func_code",
    "func_globals",
}

BLOCKED_CALLS = {
    "compile",
    "eval",
    "exec",
    "globals",
    "input",
    "locals",
    "open",
}


class SecurityError(Exception):
    """Signale une violation de la politique de securite CyberForge."""


class SecurityScanner(ast.NodeVisitor):
    """Analyse statiquement le code avant execution."""

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Bloque l'acces aux attributs introspectifs dangereux."""
        if node.attr in DANGEROUS_ATTRS:
            raise SecurityError(f"Acces interdit a l'attribut '{node.attr}'.")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Bloque l'acces direct a `__builtins__`."""
        if node.id == "__builtins__":
            raise SecurityError("Acces interdit a '__builtins__'.")
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        """Autorise uniquement une liste reduite de modules."""
        for alias in node.names:
            root_name = alias.name.split(".", maxsplit=1)[0]
            if root_name not in SAFE_MODULES:
                raise SecurityError(f"Import interdit: '{alias.name}'.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Controle les imports `from ... import ...`."""
        module_name = (node.module or "").split(".", maxsplit=1)[0]
        if module_name not in SAFE_MODULES:
            raise SecurityError(f"Import interdit: '{node.module}'.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Bloque certaines primitives dangereuses meme si elles etaient exposees."""
        if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
            raise SecurityError(f"Appel interdit: '{node.func.id}'.")
        self.generic_visit(node)


class SafeLogger:
    """Expose un logger minimal sans introspection du logger Python reel."""

    def info(self, message: str) -> None:
        """Journalise un message informatif."""
        logger.info(message)

    def error(self, message: str) -> None:
        """Journalise un message d'erreur."""
        logger.error(message)

    def warning(self, message: str) -> None:
        """Journalise un avertissement."""
        logger.warning(message)

    def debug(self, message: str) -> None:
        """Journalise un message de debug."""
        logger.debug(message)


def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Autorise uniquement les imports declares comme surs."""
    root_name = name.split(".", maxsplit=1)[0]
    if root_name in SAFE_MODULES:
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Import restreint par la politique CyberForge: '{name}'.")


SAFE_BUILTINS = {
    "__import__": guarded_import,
    "Exception": Exception,
    "AttributeError": AttributeError,
    "ImportError": ImportError,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "NameError": NameError,
    "RuntimeError": RuntimeError,
    "SyntaxError": SyntaxError,
    "TypeError": TypeError,
    "ValueError": ValueError,
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
    "format": format,
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
    "zip": zip,
}


class CyberForge:
    """Execute et trace des scripts Python dans un environnement restreint."""

    def __init__(self) -> None:
        """Initialise l'historique des executions."""
        self.history: list[dict[str, Any]] = []

    def _validate_context(self, context: dict[str, Any]) -> None:
        """Verifie que le contexte ne contient que des types simples.

        Args:
            context (dict[str, Any]): Variables injectees dans l'execution.

        Raises:
            SecurityError: Si une valeur du contexte a un type interdit.
        """
        allowed_types = (str, int, float, bool, list, dict, tuple, set, type(None))
        for key, value in context.items():
            if not isinstance(value, allowed_types):
                raise SecurityError(
                    f"Variable de contexte '{key}' avec type interdit '{type(value).__name__}'.",
                )

    def forge_and_test(
        self,
        script_name: str,
        code: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute un script dans un environnement Python restreint.

        Args:
            script_name (str): Nom logique du script.
            code (str): Code source a executer.
            context (dict[str, Any] | None): Variables simples injectees.

        Returns:
            dict[str, Any]: Resultat d'execution avec sortie et erreur eventuelle.
        """
        logger.info("CyberForge: demarrage du script '%s'.", script_name)
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        success = False
        output = ""
        error = None

        try:
            syntax_tree = ast.parse(code, filename=script_name)
            SecurityScanner().visit(syntax_tree)

            exec_globals = {"__builtins__": SAFE_BUILTINS.copy()}
            if context:
                self._validate_context(context)
                exec_globals.update(context)
            exec_globals.update({"__name__": "__cyberforge__", "logger": SafeLogger()})

            compiled_code = compile(syntax_tree, script_name, "exec")
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                exec(compiled_code, exec_globals)

            success = True
            output = stdout_capture.getvalue()
            logger.info("CyberForge: script '%s' execute avec succes.", script_name)
        except Exception:
            error = traceback.format_exc()
            logger.error("CyberForge: erreur sur le script '%s': %s", script_name, error)
        finally:
            stderr_text = stderr_capture.getvalue()
            stdout_capture.close()
            stderr_capture.close()

        result = {
            "script_name": script_name,
            "success": success,
            "output": output,
            "stderr": stderr_text,
            "error": error,
        }
        self.history.append(result)
        return result

    def get_forge_history(self) -> list[dict[str, Any]]:
        """Retourne l'historique des executions CyberForge."""
        return self.history
