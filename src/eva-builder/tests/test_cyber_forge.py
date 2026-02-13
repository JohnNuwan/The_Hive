import pytest
from eva_builder.cyber_forge import CyberForge

def test_forge_and_test_success():
    """Verify that a valid script executes successfully and captures output."""
    forge = CyberForge()
    code = "print('Hello World')"
    result = forge.forge_and_test("test_success", code)

    assert result["success"] is True, f"Expected success but got error: {result.get('error')}"
    assert "Hello World" in result["output"]
    assert result["error"] is None

def test_forge_and_test_exception():
    """Verify that a script raising an exception is handled correctly."""
    forge = CyberForge()
    code = "raise ValueError('Test Error')"
    result = forge.forge_and_test("test_exception", code)

    assert result["success"] is False
    assert result["error"] is not None
    assert "ValueError: Test Error" in result["error"]

def test_forge_and_test_syntax_error():
    """Verify that a script with syntax error is handled correctly."""
    forge = CyberForge()
    code = "print('Unclosed string"
    result = forge.forge_and_test("test_syntax", code)

    assert result["success"] is False
    assert result["error"] is not None
    assert "SyntaxError" in result["error"]

def test_forge_and_test_context_variables():
    """Verify that context variables are passed correctly."""
    forge = CyberForge()
    code = "print(f'Hello {name}')"
    context = {"name": "Universe"}
    result = forge.forge_and_test("test_context", code, context=context)

    assert result["success"] is True, f"Expected success but got error: {result.get('error')}"
    assert "Hello Universe" in result["output"]
