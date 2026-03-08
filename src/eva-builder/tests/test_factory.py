"""Tests du pipeline de generation de `CodeFactoryService`."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from eva_builder.cyber_forge import CyberForge
from eva_builder.services.api_catalog import PublicApiCatalogService
from eva_builder.services.factory import CodeFactoryService, CodeRequest


def test_generate_code_cree_les_artefacts_et_le_rapport(tmp_path: Path, monkeypatch) -> None:
    """Verifie qu'une generation Python produit les artefacts et la validation."""
    api_catalog = PublicApiCatalogService(cache_path=tmp_path / "catalog.json", source_url="https://example.test")
    service = CodeFactoryService(forge=CyberForge(), api_catalog=api_catalog)
    service.output_dir = tmp_path
    responses = iter(
        [
            "# PRD\n\nApplication de test.",
            "# DAT\n\nArchitecture simple.",
            "```python\nprint('ok builder')\n```",
        ]
    )
    prompts: list[str] = []

    async def fake_call_llm(system_prompt: str, user_prompt: str, session) -> str:
        """Retourne un flux LLM deterministe pour le test."""
        del system_prompt, session
        prompts.append(user_prompt)
        return next(responses)

    async def fake_git_commit(commit_message: str) -> str:
        """Simule un commit Git sans interaction externe."""
        return f"ignore:{commit_message}"

    monkeypatch.setattr(service, "_call_llm", fake_call_llm)
    monkeypatch.setattr(service, "_safe_git_commit", fake_git_commit)
    monkeypatch.setattr(
        service.api_catalog,
        "recommend_for_prompt",
        lambda prompt, limit: [
            {
                "name": "Alpha Vantage",
                "category": "Finance",
                "auth": "apiKey",
                "https": True,
                "cors": "Yes",
                "url": "https://www.alphavantage.co/",
                "description": "Market data and indicators",
            }
        ],
    )

    result = asyncio.run(
        service.generate_code(
            CodeRequest(
                prompt="Construis un utilitaire de demonstration.",
                filename="app.py",
                language="python",
            )
        )
    )

    assert result["status"] == "success"
    assert result["validation"] is not None
    assert result["validation"]["success"] is True
    assert "ok builder" in result["validation"]["output"]
    assert result["api_suggestions"][0]["name"] == "Alpha Vantage"

    files = result["files"]
    assert Path(files["prd"]).exists()
    assert Path(files["architecture"]).exists()
    assert Path(files["code"]).read_text(encoding="utf-8") == "print('ok builder')"

    report = json.loads(Path(files["report"]).read_text(encoding="utf-8"))
    assert report["status"] == "success"
    assert report["validation"]["success"] is True
    assert report["api_suggestions"][0]["name"] == "Alpha Vantage"
    assert any("Alpha Vantage" in prompt for prompt in prompts)


def test_generate_code_signale_un_warning_si_validation_python_echoue(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verifie qu'un code Python invalide remonte un warning au lieu d'un faux succes."""
    api_catalog = PublicApiCatalogService(cache_path=tmp_path / "catalog.json", source_url="https://example.test")
    service = CodeFactoryService(forge=CyberForge(), api_catalog=api_catalog)
    service.output_dir = tmp_path
    responses = iter(
        [
            "# PRD\n\nApplication de test.",
            "# DAT\n\nArchitecture simple.",
            "```python\nimport os\n```",
        ]
    )

    async def fake_call_llm(system_prompt: str, user_prompt: str, session) -> str:
        """Retourne un flux LLM deterministe pour le test."""
        del system_prompt, user_prompt, session
        return next(responses)

    async def fake_git_commit(commit_message: str) -> str:
        """Simule un commit Git sans interaction externe."""
        return f"ignore:{commit_message}"

    monkeypatch.setattr(service, "_call_llm", fake_call_llm)
    monkeypatch.setattr(service, "_safe_git_commit", fake_git_commit)
    monkeypatch.setattr(service.api_catalog, "recommend_for_prompt", lambda prompt, limit: [])

    result = asyncio.run(
        service.generate_code(
            CodeRequest(
                prompt="Construis un utilitaire de demonstration.",
                filename="app.py",
                language="python",
            )
        )
    )

    assert result["status"] == "warning"
    assert result["validation"] is not None
    assert result["validation"]["success"] is False
    assert "Import interdit" in result["validation"]["error"]

    report = json.loads(Path(result["files"]["report"]).read_text(encoding="utf-8"))
    assert report["status"] == "warning"
    assert report["validation"]["success"] is False
