"""
Tests unitaires pour le module Shared
Vérification des modèles Pydantic et de la configuration
"""

import pytest
from datetime import datetime
from shared import AgentStatus, HardwareMetrics, GPUMetrics, get_settings

def test_hardware_metrics_validation():
    """Vérifie que le modèle HardwareMetrics valide correctement les données"""
    gpu = GPUMetrics(
        temperature_celsius=55.5,
        utilization_percent=10.0,
        memory_used_mb=1000,
        memory_total_mb=8192
    )
    
    metrics = HardwareMetrics(
        timestamp=datetime.now(),
        cpu_percent=25.0,
        ram_used_gb=4.5,
        ram_total_gb=16.0,
        gpu=gpu
    )
    
    assert metrics.cpu_percent == 25.0
    assert metrics.ram_used_gb == 4.5
    assert metrics.gpu.temperature_celsius == 55.5

def test_agent_status_validation():
    """Vérifie le modèle AgentStatus"""
    status = AgentStatus(
        agent_id="eva-core",
        status="online",
        last_seen=datetime.now(),
        version="0.1.0"
    )
    
    assert status.status == "online"
    assert status.agent_id == "eva-core"

def test_settings_load():
    """Vérifie que les réglages se chargent (ou utilisent les défauts)"""
    settings = get_settings()
    assert settings.app_name == "THE HIVE"
    # Par défaut Redis est localhost dans les settings de base
    assert hasattr(settings, "redis_url")
