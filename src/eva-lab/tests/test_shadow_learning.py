import os
import asyncio
import json
import pytest
import shutil
from eva_lab.shadow_learning import ShadowBuffer, ShadowLearningService, Transition

@pytest.fixture
def temp_dir():
    dir_path = "test_shadow_data"
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    yield dir_path
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

@pytest.mark.asyncio
async def test_shadow_buffer_flush(temp_dir):
    buffer = ShadowBuffer(max_size=100)

    # Add items
    for i in range(10):
        buffer.add(Transition(
            observation={"val": i},
            action={"type": "TEST"},
            reward=float(i)
        ))

    assert buffer.size == 10

    # Check if flush_to_disk is async (for future compatibility)
    if asyncio.iscoroutinefunction(buffer.flush_to_disk):
        count = await buffer.flush_to_disk(temp_dir)
    else:
        count = buffer.flush_to_disk(temp_dir)

    assert count == 10
    assert buffer.size == 0

    # Verify file
    files = os.listdir(temp_dir)
    assert len(files) == 1

    filepath = os.path.join(temp_dir, files[0])
    with open(filepath, "r") as f:
        lines = f.readlines()
        assert len(lines) == 10
        data = json.loads(lines[0])
        assert data["observation"]["val"] == 0

@pytest.mark.asyncio
async def test_shadow_service_manual_flush(temp_dir):
    service = ShadowLearningService(data_dir=temp_dir, buffer_size=100)

    service.record_trade("TEST", "BUY", 100.0, 1.0, 0.0)
    assert service.buffer.size == 1

    # Manual flush
    if asyncio.iscoroutinefunction(service.manual_flush):
        count = await service.manual_flush()
    else:
        count = service.manual_flush()

    assert count == 1
    assert service.buffer.size == 0
    assert len(os.listdir(temp_dir)) == 1

@pytest.mark.asyncio
async def test_shadow_buffer_circularity(temp_dir):
    buffer = ShadowBuffer(max_size=5)
    for i in range(10):
        buffer.add(Transition(metadata={"id": i}))

    assert buffer.size == 5

    if asyncio.iscoroutinefunction(buffer.flush_to_disk):
        await buffer.flush_to_disk(temp_dir)
    else:
        buffer.flush_to_disk(temp_dir)

    files = os.listdir(temp_dir)
    filepath = os.path.join(temp_dir, files[0])
    with open(filepath, "r") as f:
        lines = f.readlines()
        assert len(lines) == 5
        last_item = json.loads(lines[-1])
        assert last_item["metadata"]["id"] == 9
