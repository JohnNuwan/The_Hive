"""Tests du bootstrap TimeScaleDB et des garde-fous de volumetrie."""

from __future__ import annotations

from contextlib import contextmanager
from unittest import TestCase
from unittest.mock import patch

from eva_lab import timescale_store


def _build_settings(*, soft_limit_gb: float = 120.0, hard_limit_gb: float = 150.0) -> dict[str, object]:
    """Construit une configuration TimeScaleDB minimale pour les tests.

    Args:
        soft_limit_gb (float): Seuil souple en Go.
        hard_limit_gb (float): Seuil dur en Go.

    Returns:
        dict[str, object]: Configuration canonique exploitable par le socle.
    """

    return {
        "enabled": True,
        "host": "timescaledb",
        "port": 5432,
        "database": "thehive",
        "user": "eva",
        "password": "secret",
        "sslmode": "prefer",
        "storage_profile": "balanced",
        "allowed_write_timeframes": ("M5", "H1", "D1"),
        "soft_limit_gb": soft_limit_gb,
        "hard_limit_gb": hard_limit_gb,
        "soft_limit_bytes": timescale_store._bytes_from_gb(soft_limit_gb),
        "hard_limit_bytes": timescale_store._bytes_from_gb(hard_limit_gb),
        "tables": timescale_store.DEFAULT_TABLES,
        "bars_table": timescale_store.DEFAULT_TABLES.bars,
        "features_table": timescale_store.DEFAULT_TABLES.features,
        "datasets_table": timescale_store.DEFAULT_TABLES.datasets,
        "arena_table": timescale_store.DEFAULT_TABLES.arena,
    }


class TimescaleStorageGuardsTests(TestCase):
    """Verifie la reparation TimeScaleDB et les limites d'ecriture OHLC."""

    def setUp(self) -> None:
        """Reinitialise l'etat global memoire avant chaque test."""

        timescale_store._set_last_bootstrap_error(None)

    def test_missing_database_is_repaired_in_place(self) -> None:
        """Cree automatiquement la base applicative absente sans reset."""

        admin_connection = object()
        data_connection = object()

        @contextmanager
        def fake_connect(database: str | None = None, **_kwargs):
            yield admin_connection if database == "postgres" else data_connection

        with patch.object(timescale_store, "get_timescale_settings", return_value=_build_settings()), \
            patch.object(timescale_store, "_connect", side_effect=fake_connect), \
            patch.object(timescale_store, "_database_exists", return_value=False), \
            patch.object(timescale_store, "_create_database") as create_database, \
            patch.object(timescale_store, "_ensure_extension") as ensure_extension, \
            patch.object(timescale_store, "_extension_exists", return_value=True), \
            patch.object(timescale_store, "_ensure_schema_objects") as ensure_schema_objects, \
            patch.object(timescale_store, "_apply_storage_profile") as apply_storage_profile, \
            patch.object(timescale_store, "_schema_objects_ready", return_value=True), \
            patch.object(timescale_store, "_database_size_bytes", return_value=42_000_000):
            runtime = timescale_store.get_timescale_runtime_status(repair=True)

        create_database.assert_called_once_with(admin_connection, "thehive", "eva")
        ensure_extension.assert_called_once_with(data_connection)
        ensure_schema_objects.assert_called_once()
        apply_storage_profile.assert_called_once()
        self.assertTrue(runtime["database_exists"])
        self.assertTrue(runtime["extension_ready"])
        self.assertTrue(runtime["schema_ready"])
        self.assertTrue(runtime["ok"])
        self.assertEqual(runtime["state"], "ready")

    def test_missing_extension_is_installed_on_existing_database(self) -> None:
        """Installe l'extension TimescaleDB quand seule la base existe deja."""

        admin_connection = object()
        data_connection = object()

        @contextmanager
        def fake_connect(database: str | None = None, **_kwargs):
            yield admin_connection if database == "postgres" else data_connection

        with patch.object(timescale_store, "get_timescale_settings", return_value=_build_settings()), \
            patch.object(timescale_store, "_connect", side_effect=fake_connect), \
            patch.object(timescale_store, "_database_exists", return_value=True), \
            patch.object(timescale_store, "_create_database") as create_database, \
            patch.object(timescale_store, "_ensure_extension") as ensure_extension, \
            patch.object(timescale_store, "_extension_exists", return_value=True), \
            patch.object(timescale_store, "_ensure_schema_objects") as ensure_schema_objects, \
            patch.object(timescale_store, "_apply_storage_profile") as apply_storage_profile, \
            patch.object(timescale_store, "_schema_objects_ready", return_value=True), \
            patch.object(timescale_store, "_database_size_bytes", return_value=51_000_000):
            runtime = timescale_store.get_timescale_runtime_status(repair=True)

        create_database.assert_not_called()
        ensure_extension.assert_called_once_with(data_connection)
        ensure_schema_objects.assert_called_once()
        apply_storage_profile.assert_called_once()
        self.assertTrue(runtime["extension_ready"])
        self.assertEqual(runtime["state"], "ready")

    def test_soft_limit_marks_runtime_as_degraded(self) -> None:
        """Expose un etat degrade au-dessus du seuil souple sans bloquer la base."""

        settings = _build_settings(soft_limit_gb=1.0, hard_limit_gb=2.0)
        admin_connection = object()
        data_connection = object()
        degraded_size = timescale_store._bytes_from_gb(1.5)

        @contextmanager
        def fake_connect(database: str | None = None, **_kwargs):
            yield admin_connection if database == "postgres" else data_connection

        with patch.object(timescale_store, "get_timescale_settings", return_value=settings), \
            patch.object(timescale_store, "_connect", side_effect=fake_connect), \
            patch.object(timescale_store, "_database_exists", return_value=True), \
            patch.object(timescale_store, "_extension_exists", return_value=True), \
            patch.object(timescale_store, "_schema_objects_ready", return_value=True), \
            patch.object(timescale_store, "_database_size_bytes", return_value=degraded_size):
            runtime = timescale_store.get_timescale_runtime_status(repair=False)

        self.assertTrue(runtime["ok"])
        self.assertEqual(runtime["state"], "degraded")
        self.assertEqual(runtime["write_guard_status"]["status"], "degraded")
        self.assertTrue(runtime["write_guard_status"]["allowed"])

    def test_hard_limit_blocks_ohlc_without_disabling_metadata_runtime(self) -> None:
        """Bloque l'OHLC au hard limit sans rendre le bootstrap indisponible."""

        settings = _build_settings(soft_limit_gb=1.0, hard_limit_gb=2.0)
        admin_connection = object()
        data_connection = object()
        blocked_size = timescale_store._bytes_from_gb(2.1)

        @contextmanager
        def fake_connect(database: str | None = None, **_kwargs):
            yield admin_connection if database == "postgres" else data_connection

        with patch.object(timescale_store, "get_timescale_settings", return_value=settings), \
            patch.object(timescale_store, "_connect", side_effect=fake_connect), \
            patch.object(timescale_store, "_database_exists", return_value=True), \
            patch.object(timescale_store, "_extension_exists", return_value=True), \
            patch.object(timescale_store, "_schema_objects_ready", return_value=True), \
            patch.object(timescale_store, "_database_size_bytes", return_value=blocked_size):
            runtime = timescale_store.get_timescale_runtime_status(repair=False)

        self.assertTrue(runtime["ok"])
        self.assertEqual(runtime["state"], "write_guard_blocked")
        self.assertEqual(runtime["write_guard_status"]["status"], "blocked")
        self.assertFalse(runtime["write_guard_status"]["allowed"])

    def test_non_allowed_timeframe_is_refused_explicitly(self) -> None:
        """Refuse explicitement M1 meme si la base est par ailleurs utilisable."""

        with patch.object(
            timescale_store,
            "get_timescale_runtime_status",
            return_value={
                "state": "ready",
                "storage_profile": "balanced",
                "db_size_bytes": 42_000_000,
                "database_exists": True,
                "extension_ready": True,
                "schema_ready": True,
                "last_bootstrap_error": None,
                "allowed_write_timeframes": ["M5", "H1", "D1"],
                "write_guard_status": {
                    "allowed": True,
                    "status": "allowed",
                    "reason": "within_limits",
                    "db_size_bytes": 42_000_000,
                    "soft_limit_bytes": timescale_store._bytes_from_gb(120.0),
                    "hard_limit_bytes": timescale_store._bytes_from_gb(150.0),
                    "allowed_timeframes": ["M5", "H1", "D1"],
                },
            },
        ):
            diagnostic = timescale_store.evaluate_ohlc_write_request("M1", repair=False)

        self.assertFalse(diagnostic["allowed"])
        self.assertEqual(diagnostic["status"], "timeframe_blocked")
        self.assertEqual(diagnostic["reason"], "timeframe_non_autorise")


if __name__ == "__main__":
    import unittest

    unittest.main()
