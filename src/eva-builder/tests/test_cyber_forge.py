import pytest
from eva_builder.cyber_forge import CyberForge

def test_forge_and_test_success():
    """Test successful execution of a script."""
    forge = CyberForge()
    # Simple script that prints something
    code = "print('Hello World')"
    result = forge.forge_and_test("test_success", code)

    assert result["success"] is True
    assert "Hello World" in result["output"]
    assert result["error"] is None

def test_forge_and_test_exception():
    """Test script that raises an exception."""
    forge = CyberForge()
    # Script raising ValueError
    code = "raise ValueError('Intentional Error')"
    result = forge.forge_and_test("test_exception", code)

    assert result["success"] is False
    # The error message should contain the exception details
    assert result["error"] is not None
    assert "ValueError: Intentional Error" in result["error"]

def test_forge_and_test_syntax_error():
    """Test script with syntax error."""
    forge = CyberForge()
    # Script with syntax error (missing closing quote)
    code = "print('Hello"
    result = forge.forge_and_test("test_syntax", code)

    assert result["success"] is False
    assert result["error"] is not None
    assert "SyntaxError" in result["error"]

def test_forge_and_test_safe_import():
    """Test that safe imports are allowed."""
    forge = CyberForge()
    code = "import math; print(math.sqrt(4))"
    result = forge.forge_and_test("test_safe_import", code)

    assert result["success"] is True
    assert "2.0" in result["output"]

def test_safe_logging():
    """Test that logger works safely."""
    forge = CyberForge()
    code = "logger.info('Safe log test')"
    result = forge.forge_and_test("test_logging", code)
    assert result["success"] is True
    # Note: logger.info goes to system log, not stdout_capture unless configured.
    # But execution should succeed.
