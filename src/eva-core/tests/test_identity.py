"""
Tests for EVA Core Identity & Hardware Configuration.
"""
import sys
import os
from pathlib import Path

# Add src/eva-core/eva_core to sys.path to import identity.py directly
# avoiding the package __init__.py which triggers full app load
current_dir = Path(__file__).parent
eva_core_dir = current_dir.parent / "eva_core"
sys.path.append(str(eva_core_dir))

import identity as eva_identity

def test_identity_constants_exist():
    """Verify that identity constants are defined and not empty."""
    assert isinstance(eva_identity.EVA_CORE_IDENTITY, str)
    assert len(eva_identity.EVA_CORE_IDENTITY) > 0
    assert "EVA" in eva_identity.EVA_CORE_IDENTITY

    assert isinstance(eva_identity.EVA_GAMIFICATION_PROTOCOL, str)
    assert len(eva_identity.EVA_GAMIFICATION_PROTOCOL) > 0

def test_hardware_inventory_completeness():
    """Verify that all requested hardware items are present in the protocol."""
    inventory = eva_identity.EVA_GAMIFICATION_PROTOCOL

    # Check for specific hardware items requested by the user
    assert "AXE-BME20P1AJ04A02" in inventory
    assert "Halo" in inventory
    assert "Brilliant Labs" in inventory
    assert "3090 FE" in inventory
    assert "H100" in inventory

    # Check for gamification context (Economy)
    assert "GAMIFICATION" in inventory
    assert "ÉCONOMIE" in inventory
    assert "REVENUS" in inventory
    assert "PROFIT" in inventory
    assert "ARGENT" in inventory
    assert "CREDITS" in inventory
