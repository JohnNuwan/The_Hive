"""Tests du catalogue d'APIs publiques pour `eva-builder`."""

from __future__ import annotations

import json
from pathlib import Path

from eva_builder.services.api_catalog import PublicApiCatalogService


SAMPLE_MARKDOWN = """
## Index

### Finance
API | Description | Auth | HTTPS | CORS
|:---|:---|:---|:---|:---|
| [Alpha Vantage](https://www.alphavantage.co/) | Market data and indicators | apiKey | Yes | Yes |
| [Frankfurter](https://www.frankfurter.app/) | Exchange rates and conversions | No | Yes | Unknown |

### Machine Learning
API | Description | Auth | HTTPS | CORS
|:---|:---|:---|:---|:---|
| [Hugging Face](https://huggingface.co/inference-api) | Inference API for NLP and vision | apiKey | Yes | Unknown |
"""


def test_parse_markdown_catalog_extrait_les_entrees() -> None:
    """Verifie que le parser Markdown reconstruit correctement les APIs."""
    entries = PublicApiCatalogService._parse_markdown_catalog(SAMPLE_MARKDOWN)

    assert len(entries) == 3
    assert entries[0]["name"] == "Alpha Vantage"
    assert entries[0]["category"] == "Finance"
    assert entries[0]["https"] is True
    assert entries[2]["category"] == "Machine Learning"


def test_search_entries_filtre_et_score_depuis_le_cache(tmp_path: Path) -> None:
    """Verifie la recherche locale avec filtres et tri par pertinence."""
    cache_path = tmp_path / "public_api_catalog.json"
    entries = PublicApiCatalogService._parse_markdown_catalog(SAMPLE_MARKDOWN)
    cache_path.write_text(
        json.dumps(
            {
                "status": "success",
                "source_url": "https://example.test/readme.md",
                "entries": entries,
                "categories": ["Finance", "Machine Learning"],
                "total_entries": len(entries),
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    service = PublicApiCatalogService(cache_path=cache_path, source_url="https://example.test")
    finance_results = service.search_entries(query="market data", category="Finance", https_only=True)
    ml_results = service.recommend_for_prompt("Je veux un SaaS NLP avec inference vision")

    assert len(finance_results) == 1
    assert finance_results[0]["name"] == "Alpha Vantage"
    assert len(ml_results) == 1
    assert ml_results[0]["name"] == "Hugging Face"
