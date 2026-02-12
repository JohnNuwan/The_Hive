import pytest
from eva_builder.cyber_forge import CyberForge

def test_logger_escape():
    """Test escaping via logger object."""
    forge = CyberForge()
    code = """
try:
    if 'logger' in locals() or 'logger' in globals():
        os_mod = logger.__class__.__init__.__globals__['os']
        print(f"HACKED: {os_mod.name}")
    else:
        # Try to use 'log' function
        os_mod = log.__init__.__globals__['os']
        print(f"HACKED: {os_mod.name}")
except Exception as e:
    print(f"FAILED: {e}")
"""
    result = forge.forge_and_test("test_logger_escape", code)
    assert "HACKED" not in result["output"]
    # Check if blocked (either exception caught inside script or outside)
    is_blocked = (
        result["success"] is False or
        "FAILED" in result["output"]
    )
    assert is_blocked

def test_subclass_traversal_escape():
    """Test escaping via object subclasses."""
    forge = CyberForge()
    code = """
try:
    classes = ().__class__.__base__.__subclasses__()
    print(f"Classes found: {len(classes)}")
except Exception as e:
    print(f"FAILED: {e}")
"""
    result = forge.forge_and_test("test_subclass_traversal", code)
    assert result["success"] is False
    assert "SecurityError" in result["error"]

def test_getattr_escape():
    """Test using getattr to access hidden attributes."""
    forge = CyberForge()
    code = """
try:
    cls = getattr((), "__class__")
    print(f"Got class: {cls}")
except Exception as e:
    print(f"FAILED: {e}")
"""
    result = forge.forge_and_test("test_getattr_escape", code)

    if result["success"]:
        assert "FAILED" in result["output"]
        assert "name 'getattr' is not defined" in result["output"]
    else:
        assert "NameError" in result["error"]

def test_import_os_blocked():
    """Test that importing 'os' is blocked."""
    forge = CyberForge()
    code = "import os; print(os.name)"
    result = forge.forge_and_test("test_import_os", code)
    # Blocked by ImportError from guarded_import
    assert result["success"] is False
    assert "ImportError" in result["error"]

def test_open_blocked():
    """Test that 'open' is blocked."""
    forge = CyberForge()
    code = "f = open('test_hacked.txt', 'w')"
    result = forge.forge_and_test("test_open", code)
    # Blocked by NameError (open removed from builtins)
    assert result["success"] is False
    assert "NameError" in result["error"]

def test_builtin_import_blocked():
    """Test accessing __import__ directly with dangerous module."""
    forge = CyberForge()
    code = "__import__('os')"
    result = forge.forge_and_test("test_builtin_import", code)
    assert result["success"] is False
    assert "ImportError" in result["error"]
