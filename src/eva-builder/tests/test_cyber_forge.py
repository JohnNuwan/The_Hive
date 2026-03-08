"""Tests de securite et d'execution pour CyberForge."""

from eva_builder.cyber_forge import CyberForge


def test_forge_and_test_success() -> None:
    """Verifie qu'un script valide s'execute et produit une sortie."""
    forge = CyberForge()
    result = forge.forge_and_test("test_success", "print('Hello World')")

    assert result["success"] is True
    assert "Hello World" in result["output"]
    assert result["error"] is None


def test_forge_and_test_exception() -> None:
    """Verifie qu'une exception utilisateur est remontee proprement."""
    forge = CyberForge()
    result = forge.forge_and_test("test_exception", "raise ValueError('Test Error')")

    assert result["success"] is False
    assert result["error"] is not None
    assert "ValueError: Test Error" in result["error"]


def test_forge_and_test_syntax_error() -> None:
    """Verifie qu'une erreur de syntaxe est capturee par le resultat."""
    forge = CyberForge()
    result = forge.forge_and_test("test_syntax", "print('Unclosed string")

    assert result["success"] is False
    assert result["error"] is not None
    assert "SyntaxError" in result["error"]


def test_forge_and_test_context_variables() -> None:
    """Verifie que le contexte simple est accessible dans le script."""
    forge = CyberForge()
    result = forge.forge_and_test(
        "test_context",
        "print(f'Hello {name}')",
        context={"name": "Universe"},
    )

    assert result["success"] is True
    assert "Hello Universe" in result["output"]


def test_forge_blocke_import_interdit() -> None:
    """Verifie qu'un import non autorise est refuse avant execution."""
    forge = CyberForge()
    result = forge.forge_and_test("test_import", "import os\nprint('x')")

    assert result["success"] is False
    assert result["error"] is not None
    assert "Import interdit" in result["error"]


def test_forge_blocke_appel_interdit() -> None:
    """Verifie qu'un appel dangereux reste bloque meme si expose."""
    forge = CyberForge()
    result = forge.forge_and_test("test_open", "open('secret.txt', 'w')")

    assert result["success"] is False
    assert result["error"] is not None
    assert "Appel interdit" in result["error"]
