"""Shared test fixtures and hardware mocking for CI environments.

Injects fake picamera2/Hailo modules into sys.modules so bird_monitor
can be imported without the actual hardware SDKs installed.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest

# --- Mock hardware modules before any test imports bird_monitor ---

# Create fake picamera2 package
_picamera2_mod = types.ModuleType("picamera2")
_picamera2_mod.Picamera2 = MagicMock()

_devices_mod = types.ModuleType("picamera2.devices")
_hailo_mod = types.ModuleType("picamera2.devices.Hailo")
_hailo_mod.Hailo = MagicMock()
_devices_mod.Hailo = _hailo_mod

sys.modules.setdefault("picamera2", _picamera2_mod)
sys.modules.setdefault("picamera2.devices", _devices_mod)
sys.modules.setdefault("picamera2.devices.Hailo", _hailo_mod)

# Now safe to import bird_monitor
from bird_monitor import BirdDB, CentroidTracker  # noqa: E402


@pytest.fixture
def in_memory_db(tmp_path):
    """BirdDB backed by a temp file (cleaned up automatically)."""
    db_path = str(tmp_path / "test_birds.db")
    return BirdDB(db_path)


@pytest.fixture
def tracker():
    """CentroidTracker with small thresholds for easy testing."""
    return CentroidTracker(max_disappeared=3, max_distance=100)
