
import pytest
from eva_builder.cyber_forge import CyberForge

def test_format_builtin_escape():
    """Test escaping via format() built-in function."""
    forge = CyberForge()
    # If format() is available and allows access to attributes
    # But since we remove it from SAFE_BUILTINS, it should raise NameError
    code = """
try:
    print(format(1, 'd'))
except NameError:
    print("NameError: format not defined")
except Exception as e:
    print(f"FAILED: {e}")
"""
    result = forge.forge_and_test("test_format_builtin", code)
    if result["success"]:
        assert "NameError: format not defined" in result["output"]
    else:
        # It might fail with NameError in error field
        assert "NameError" in result["error"]

def test_str_format_method_escape():
    """Test escaping via str.format() method."""
    forge = CyberForge()
    # This uses str.format which is a method on string objects
    # We expect this to be blocked by blocking 'format' attribute access
    code = """
try:
    # Attempt to access info.__func__.__globals__ of the logger object using format
    secret = "{0.info.__func__.__globals__}".format(logger)
    print("EXPLOIT SUCCESSFUL: Accessed globals")
except Exception as e:
    print(f"FAILED: {e}")
"""
    result = forge.forge_and_test("test_str_format_escape", code)

    # It should fail during AST analysis with SecurityError because we blocked 'format' attribute
    if result["success"]:
        # If it ran, it must have failed with an exception inside or printed FAILED
        assert "EXPLOIT SUCCESSFUL" not in result["output"]
        assert "SecurityError" in result["output"] or "FAILED" in result["output"]
    else:
        # Expected behavior: SecurityError raised during scanning
        assert "SecurityError" in result["error"]
        assert "Access to attribute 'format' is forbidden" in result["error"]

def test_format_map_escape():
    """Test escaping via str.format_map() method."""
    forge = CyberForge()
    code = """
try:
    # format_map is similar to format
    "{x}".format_map({'x': 1})
    print("format_map executed")
except Exception as e:
    print(f"FAILED: {e}")
"""
    result = forge.forge_and_test("test_format_map_escape", code)

    # Should be blocked
    if result["success"]:
        # If execution succeeded, we failed to block it.
        # Unless it printed FAILED (which means it was blocked or errored).
        # We want to ensure it did NOT execute successfully.
        assert "format_map executed" not in result["output"]
        assert "SecurityError" in result["output"] or "FAILED" in result["output"]
    else:
        # Expected behavior: SecurityError raised during scanning
        assert "SecurityError" in result["error"]
        assert "Access to attribute 'format_map' is forbidden" in result["error"]
