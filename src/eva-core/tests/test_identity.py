"""
Tests for EVA Core Identity & Hardware Configuration.
"""

from eva_core.identity import EVA_CORE_IDENTITY, EVA_GAMIFICATION_PROTOCOL

def test_identity_constants_exist():
    """Verify that identity constants are defined and not empty."""
    assert isinstance(EVA_CORE_IDENTITY, str)
    assert len(EVA_CORE_IDENTITY) > 0
    assert "EVA" in EVA_CORE_IDENTITY

    assert isinstance(EVA_GAMIFICATION_PROTOCOL, str)
    assert len(EVA_GAMIFICATION_PROTOCOL) > 0

def test_hardware_inventory_completeness():
    """Verify that all requested hardware items are present in the protocol."""
    inventory = EVA_GAMIFICATION_PROTOCOL

    # Check for specific hardware items requested by the user
    assert "AXE-BME20P1AJ04A02" in inventory
    assert "Halo" in inventory
    assert "Brilliant Labs" in inventory
    assert "3090 FE" in inventory
    assert "H100" in inventory

    # Check for gamification context
    assert "GAMIFICATION" in inventory
    assert "Hardware" in inventory
