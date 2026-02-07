"""
Docker & System Monitoring Service — THE HIVE
Provides real-time system metrics (psutil) and Docker container stats.
"""

import asyncio
import logging
import platform
import time
from typing import Any

logger = logging.getLogger(__name__)

# ═══ Conditional Imports ═══
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("⚠️ psutil not available — system metrics will be simulated")

try:
    import docker as docker_sdk

    DOCKER_SDK_AVAILABLE = True
except ImportError:
    DOCKER_SDK_AVAILABLE = False
    logger.warning("⚠️ docker SDK not available — will use subprocess fallback")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER — Parse size strings
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_size_to_mb(s: str) -> float:
    """Parse '50.32MiB', '1.2GiB', etc. → MB"""
    s = s.strip()
    multipliers = {
        "TiB": 1024 * 1024,
        "TB": 1024 * 1024,
        "GiB": 1024,
        "GB": 1024,
        "MiB": 1,
        "MB": 1,
        "KiB": 1 / 1024,
        "KB": 1 / 1024,
        "kB": 1 / 1024,
        "B": 1 / (1024 * 1024),
    }
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            return float(s[: -len(suffix)].strip()) * mult
    try:
        return float(s)
    except ValueError:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM MONITOR
# ═══════════════════════════════════════════════════════════════════════════════


class SystemMonitor:
    """Monitors host resources and Docker containers."""

    def __init__(self):
        self._docker_client = None
        self._boot_time = time.time()
        self._last_net_io = None
        self._last_net_time = 0.0
        self._last_disk_io = None
        self._last_disk_time = 0.0
        self._init_docker()

    def _init_docker(self):
        if DOCKER_SDK_AVAILABLE:
            try:
                self._docker_client = docker_sdk.from_env()
                self._docker_client.ping()
                logger.info("✅ Docker client connected")
            except Exception as e:
                logger.warning(f"⚠️ Docker client init failed: {e}")
                self._docker_client = None

    # ═══════════════════════════════════════════════════════════════════════
    # SYSTEM METRICS
    # ═══════════════════════════════════════════════════════════════════════

    async def get_system_metrics(self) -> dict[str, Any]:
        if not PSUTIL_AVAILABLE:
            return self._simulated_metrics()

        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_freq = psutil.cpu_freq()
            cpu_count = psutil.cpu_count(logical=True) or 4

            cpu_model = platform.processor() or f"{cpu_count}-Core Processor"

            # CPU temperature
            cpu_temp = 0.0
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for _, entries in temps.items():
                        if entries:
                            cpu_temp = entries[0].current
                            break
            except (AttributeError, Exception):
                pass

            # Memory
            mem = psutil.virtual_memory()

            # Disk
            disk = psutil.disk_usage("/")
            disk_read_speed = 0.0
            disk_write_speed = 0.0
            try:
                disk_io = psutil.disk_io_counters()
                now = time.time()
                if self._last_disk_io and (now - self._last_disk_time) > 0:
                    dt = now - self._last_disk_time
                    disk_read_speed = (
                        disk_io.read_bytes - self._last_disk_io.read_bytes
                    ) / dt / (1024 * 1024)
                    disk_write_speed = (
                        disk_io.write_bytes - self._last_disk_io.write_bytes
                    ) / dt / (1024 * 1024)
                self._last_disk_io = disk_io
                self._last_disk_time = now
            except Exception:
                pass

            # Network (with delta for speed)
            net_io = psutil.net_io_counters()
            rx_speed = 0.0
            tx_speed = 0.0
            now = time.time()
            if self._last_net_io and (now - self._last_net_time) > 0:
                dt = now - self._last_net_time
                rx_speed = (
                    net_io.bytes_recv - self._last_net_io.bytes_recv
                ) / dt / (1024 * 1024)
                tx_speed = (
                    net_io.bytes_sent - self._last_net_io.bytes_sent
                ) / dt / (1024 * 1024)
            self._last_net_io = net_io
            self._last_net_time = now

            # Uptime
            boot_time = psutil.boot_time()
            uptime = time.time() - boot_time

            # GPU
            gpu_info = await self._get_gpu_info()

            return {
                "cpu": {
                    "usage": round(cpu_percent, 1),
                    "cores": cpu_count,
                    "model": cpu_model[:50],
                    "temp": round(cpu_temp, 0),
                    "freq": round(cpu_freq.current, 0) if cpu_freq else 0,
                },
                "memory": {
                    "used": round(mem.used / (1024**3), 1),
                    "total": round(mem.total / (1024**3), 1),
                    "percent": round(mem.percent, 1),
                },
                "gpu": gpu_info,
                "disk": {
                    "used": round(disk.used / (1024**3), 0),
                    "total": round(disk.total / (1024**3), 0),
                    "percent": round(disk.percent, 1),
                    "read_speed": round(max(disk_read_speed, 0), 1),
                    "write_speed": round(max(disk_write_speed, 0), 1),
                },
                "network": {
                    "rx_bytes": net_io.bytes_recv,
                    "tx_bytes": net_io.bytes_sent,
                    "rx_speed": round(max(rx_speed, 0), 1),
                    "tx_speed": round(max(tx_speed, 0), 1),
                },
                "uptime": round(uptime),
                "platform": platform.system(),
                "hostname": platform.node(),
                "real_data": True,
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return self._simulated_metrics()

    async def _get_gpu_info(self) -> dict | None:
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0 and stdout:
                parts = stdout.decode().strip().split(",")
                if len(parts) >= 5:
                    return {
                        "name": parts[0].strip(),
                        "usage": float(parts[1].strip()),
                        "memory_used": round(float(parts[2].strip()) / 1024, 1),
                        "memory_total": round(float(parts[3].strip()) / 1024, 1),
                        "temp": float(parts[4].strip()),
                    }
        except (FileNotFoundError, Exception):
            pass
        return None

    # ═══════════════════════════════════════════════════════════════════════
    # DOCKER CONTAINERS
    # ═══════════════════════════════════════════════════════════════════════

    async def get_docker_containers(self) -> list[dict[str, Any]]:
        # Method 1: Docker SDK
        if self._docker_client:
            try:
                return await self._get_containers_sdk()
            except Exception as e:
                logger.warning(f"Docker SDK failed: {e}")

        # Method 2: subprocess
        try:
            return await self._get_containers_subprocess()
        except Exception as e:
            logger.warning(f"Docker subprocess failed: {e}")

        return []

    async def _get_containers_sdk(self) -> list[dict]:
        loop = asyncio.get_event_loop()
        docker_containers = await loop.run_in_executor(
            None, lambda: self._docker_client.containers.list(all=True)
        )

        containers = []
        for c in docker_containers:
            try:
                if c.status == "running":
                    stats = await loop.run_in_executor(
                        None, lambda cont=c: cont.stats(stream=False)
                    )

                    # CPU percentage
                    cpu_percent = 0.0
                    cpu_stats = stats.get("cpu_stats", {})
                    precpu_stats = stats.get("precpu_stats", {})
                    cpu_delta = cpu_stats.get("cpu_usage", {}).get(
                        "total_usage", 0
                    ) - precpu_stats.get("cpu_usage", {}).get("total_usage", 0)
                    system_delta = cpu_stats.get(
                        "system_cpu_usage", 0
                    ) - precpu_stats.get("system_cpu_usage", 0)
                    percpu = cpu_stats.get("cpu_usage", {}).get("percpu_usage", [1])
                    num_cpus = len(percpu) if percpu else 1
                    if system_delta > 0 and cpu_delta > 0:
                        cpu_percent = (cpu_delta / system_delta) * num_cpus * 100.0

                    # Memory
                    mem_stats = stats.get("memory_stats", {})
                    mem_usage = mem_stats.get("usage", 0)
                    mem_limit = mem_stats.get("limit", 1)
                    mem_percent = (mem_usage / mem_limit) * 100 if mem_limit > 0 else 0

                    # Network
                    net_stats = stats.get("networks", {})
                    rx_bytes = sum(v.get("rx_bytes", 0) for v in net_stats.values())
                    tx_bytes = sum(v.get("tx_bytes", 0) for v in net_stats.values())

                    # PIDs
                    pids = stats.get("pids_stats", {}).get("current", 0)
                else:
                    cpu_percent = 0.0
                    mem_usage = 0
                    mem_limit = 0
                    mem_percent = 0.0
                    rx_bytes = 0
                    tx_bytes = 0
                    pids = 0

                # Status mapping
                status_map = {
                    "running": "running",
                    "exited": "stopped",
                    "restarting": "restarting",
                    "paused": "paused",
                    "created": "stopped",
                    "dead": "stopped",
                }

                # Uptime calculation
                started_at = c.attrs.get("State", {}).get("StartedAt", "")
                uptime_str = ""
                if started_at and c.status == "running":
                    try:
                        # Docker timestamps like "2026-02-07T15:26:38.123456Z"
                        start_ts = started_at.replace("Z", "+00:00")
                        from datetime import datetime, timezone

                        start_dt = datetime.fromisoformat(start_ts)
                        delta = datetime.now(timezone.utc) - start_dt
                        hours = int(delta.total_seconds() // 3600)
                        minutes = int((delta.total_seconds() % 3600) // 60)
                        uptime_str = f"{hours}h {minutes}m"
                    except Exception:
                        uptime_str = started_at[:19]

                containers.append(
                    {
                        "id": c.short_id,
                        "name": c.name,
                        "status": status_map.get(c.status, "stopped"),
                        "cpu_percent": round(cpu_percent, 2),
                        "memory_usage": round(mem_usage / (1024 * 1024), 1),
                        "memory_limit": round(mem_limit / (1024 * 1024), 0),
                        "memory_percent": round(mem_percent, 1),
                        "network_rx": rx_bytes,
                        "network_tx": tx_bytes,
                        "pids": pids or 0,
                        "image": (
                            c.image.tags[0]
                            if c.image.tags
                            else str(c.image.short_id)
                        ),
                        "uptime": uptime_str,
                    }
                )
            except Exception as e:
                logger.debug(f"Error getting stats for {c.name}: {e}")
                containers.append(
                    {
                        "id": c.short_id,
                        "name": c.name,
                        "status": "stopped" if c.status != "running" else "running",
                        "cpu_percent": 0,
                        "memory_usage": 0,
                        "memory_limit": 0,
                        "memory_percent": 0,
                        "network_rx": 0,
                        "network_tx": 0,
                        "pids": 0,
                        "image": (
                            c.image.tags[0] if c.image.tags else "unknown"
                        ),
                        "uptime": "",
                    }
                )

        return containers

    async def _get_containers_subprocess(self) -> list[dict]:
        """Fallback: use docker CLI via subprocess."""
        # Get all containers with their status
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "ps",
            "-a",
            "--format",
            "{{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}\t{{.State}}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_ps, _ = await proc.communicate()
        if proc.returncode != 0:
            return []

        containers_map: dict[str, dict] = {}
        for line in stdout_ps.decode().strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            cid = parts[0][:12]
            state = parts[4].lower()
            status = "running" if state == "running" else (
                "restarting" if state == "restarting" else (
                    "paused" if state == "paused" else "stopped"
                )
            )
            containers_map[cid] = {
                "id": cid,
                "name": parts[1],
                "status": status,
                "cpu_percent": 0,
                "memory_usage": 0,
                "memory_limit": 0,
                "memory_percent": 0,
                "network_rx": 0,
                "network_tx": 0,
                "pids": 0,
                "image": parts[3],
                "uptime": parts[2],
            }

        # Get stats for running containers
        proc2 = await asyncio.create_subprocess_exec(
            "docker",
            "stats",
            "--no-stream",
            "--format",
            "{{.ID}}\t{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.PIDs}}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_stats, _ = await proc2.communicate()

        if proc2.returncode == 0 and stdout_stats:
            for line in stdout_stats.decode().strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) < 7:
                    continue
                cid = parts[0][:12]
                if cid not in containers_map:
                    continue

                # CPU
                try:
                    containers_map[cid]["cpu_percent"] = float(
                        parts[2].replace("%", "").strip()
                    )
                except ValueError:
                    pass

                # Memory
                try:
                    mem_parts = parts[3].split("/")
                    containers_map[cid]["memory_usage"] = _parse_size_to_mb(
                        mem_parts[0].strip()
                    )
                    containers_map[cid]["memory_limit"] = _parse_size_to_mb(
                        mem_parts[1].strip()
                    )
                except (ValueError, IndexError):
                    pass

                try:
                    containers_map[cid]["memory_percent"] = float(
                        parts[4].replace("%", "").strip()
                    )
                except ValueError:
                    pass

                # Network
                try:
                    net_parts = parts[5].split("/")
                    containers_map[cid]["network_rx"] = int(
                        _parse_size_to_mb(net_parts[0].strip()) * 1024 * 1024
                    )
                    containers_map[cid]["network_tx"] = int(
                        _parse_size_to_mb(net_parts[1].strip()) * 1024 * 1024
                    )
                except (ValueError, IndexError):
                    pass

                # PIDs
                try:
                    containers_map[cid]["pids"] = int(parts[6].strip())
                except ValueError:
                    pass

        return list(containers_map.values())

    def _simulated_metrics(self) -> dict:
        import random

        return {
            "cpu": {
                "usage": round(random.uniform(5, 30), 1),
                "cores": 8,
                "model": "Unknown (psutil not available)",
                "temp": 0,
            },
            "memory": {"used": 8.0, "total": 16.0, "percent": 50.0},
            "gpu": None,
            "disk": {
                "used": 200,
                "total": 500,
                "percent": 40.0,
                "read_speed": 0,
                "write_speed": 0,
            },
            "network": {
                "rx_bytes": 0,
                "tx_bytes": 0,
                "rx_speed": 0,
                "tx_speed": 0,
            },
            "uptime": int(time.time() - self._boot_time),
            "real_data": False,
        }
