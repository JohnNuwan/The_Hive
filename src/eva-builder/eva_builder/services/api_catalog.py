"""Catalogue local des APIs publiques utiles a `eva-builder`."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import ClientError
from aiohttp.resolver import ThreadedResolver

logger = logging.getLogger(__name__)


class PublicApiCatalogService:
    """Synchronise et recherche un catalogue d'APIs publiques."""

    DEFAULT_SOURCE_URL = (
        "https://raw.githubusercontent.com/public-apis/public-apis/master/README.md"
    )

    def __init__(
        self,
        cache_path: Path | None = None,
        source_url: str | None = None,
    ) -> None:
        """Initialise la source distante et le cache local.

        Args:
            cache_path (Path | None): Emplacement du cache JSON local.
            source_url (str | None): URL du README servant de source.
        """
        self.cache_path = cache_path or (
            Path(os.getcwd()) / "data" / "builder" / "public_api_catalog.json"
        )
        self.source_url = source_url or os.getenv(
            "EVA_BUILDER_PUBLIC_API_SOURCE_URL",
            self.DEFAULT_SOURCE_URL,
        )
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    async def sync_catalog(self) -> dict[str, Any]:
        """Telecharge le README de reference et le transforme en cache JSON.

        Returns:
            dict[str, Any]: Resume de la synchronisation.

        Raises:
            RuntimeError: Si la source distante est inaccessible ou vide.
        """
        logger.info("Synchronisation du catalogue d'APIs publiques depuis %s", self.source_url)

        connector = aiohttp.TCPConnector(resolver=ThreadedResolver())
        try:
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(self.source_url, timeout=120) as response:
                    if response.status != 200:
                        raise RuntimeError(
                            f"Source des APIs publiques indisponible (status={response.status}).",
                        )
                    markdown = await response.text()
        except ClientError as exc:
            raise RuntimeError(
                f"Echec de telechargement du catalogue d'APIs publiques: {exc}",
            ) from exc

        entries = self._parse_markdown_catalog(markdown)
        if not entries:
            raise RuntimeError("Aucune API publique n'a pu etre extraite de la source distante.")

        categories = sorted({entry["category"] for entry in entries})
        payload = {
            "status": "success",
            "source_url": self.source_url,
            "entries": entries,
            "categories": categories,
            "total_entries": len(entries),
        }
        self.cache_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        logger.info(
            "Catalogue d'APIs publiques synchronise: %s entrees, %s categories.",
            len(entries),
            len(categories),
        )
        return {
            "status": "success",
            "source_url": self.source_url,
            "cache_path": str(self.cache_path),
            "total_entries": len(entries),
            "total_categories": len(categories),
        }

    def load_catalog(self) -> dict[str, Any]:
        """Charge le cache local du catalogue.

        Returns:
            dict[str, Any]: Catalogue local ou structure vide.
        """
        if not self.cache_path.exists():
            return {
                "status": "empty",
                "source_url": self.source_url,
                "entries": [],
                "categories": [],
                "total_entries": 0,
            }

        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Lecture du cache des APIs publiques impossible: %s", exc)
            return {
                "status": "error",
                "source_url": self.source_url,
                "entries": [],
                "categories": [],
                "total_entries": 0,
                "error": str(exc),
            }

    def search_entries(
        self,
        query: str = "",
        category: str | None = None,
        auth: str | None = None,
        https_only: bool = False,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Recherche des APIs publiques dans le cache local.

        Args:
            query (str): Texte libre pour scorer les resultats.
            category (str | None): Categorie cible.
            auth (str | None): Filtre d'authentification.
            https_only (bool): Garde uniquement les APIs HTTPS.
            limit (int): Nombre maximum de resultats.

        Returns:
            list[dict[str, Any]]: Resultats tries par pertinence.
        """
        catalog = self.load_catalog()
        entries = catalog.get("entries", [])
        if not entries:
            return []

        normalized_query = query.strip().lower()
        normalized_category = (category or "").strip().lower()
        normalized_auth = (auth or "").strip().lower()
        query_tokens = self._tokenize(normalized_query)
        scored_results: list[tuple[int, dict[str, Any]]] = []

        for entry in entries:
            if normalized_category and entry["category"].lower() != normalized_category:
                continue
            if normalized_auth and entry["auth"].lower() != normalized_auth:
                continue
            if https_only and not entry["https"]:
                continue

            score = 0
            haystack = " ".join(
                [
                    entry["name"],
                    entry["description"],
                    entry["category"],
                    entry["auth"],
                    entry["url"],
                ]
            ).lower()

            if normalized_query:
                if normalized_query in entry["name"].lower():
                    score += 8
                if normalized_query in entry["description"].lower():
                    score += 5
                for token in query_tokens:
                    if token in haystack:
                        score += 2
                if score == 0:
                    continue
            else:
                score = 1

            scored_results.append((score, entry))

        scored_results.sort(
            key=lambda item: (-item[0], item[1]["category"].lower(), item[1]["name"].lower()),
        )
        return [entry for _, entry in scored_results[: max(limit, 1)]]

    def recommend_for_prompt(self, prompt: str, limit: int = 5) -> list[dict[str, Any]]:
        """Retourne les APIs les plus pertinentes pour un brief produit.

        Args:
            prompt (str): Description fonctionnelle du produit vise.
            limit (int): Nombre maximum de suggestions.

        Returns:
            list[dict[str, Any]]: APIs recommandees pour enrichir la conception.
        """
        return self.search_entries(query=prompt, limit=limit)

    def get_stats(self) -> dict[str, Any]:
        """Retourne un resume du cache local.

        Returns:
            dict[str, Any]: Nombre d'entrees et categories disponibles.
        """
        catalog = self.load_catalog()
        return {
            "status": catalog.get("status", "unknown"),
            "cache_path": str(self.cache_path),
            "source_url": catalog.get("source_url", self.source_url),
            "total_entries": catalog.get("total_entries", 0),
            "total_categories": len(catalog.get("categories", [])),
        }

    @classmethod
    def _parse_markdown_catalog(cls, markdown: str) -> list[dict[str, Any]]:
        """Extrait les entrees d'APIs depuis le README Markdown.

        Args:
            markdown (str): Contenu brut du README.

        Returns:
            list[dict[str, Any]]: Liste d'APIs parsees.
        """
        current_category: str | None = None
        entries: list[dict[str, Any]] = []

        for raw_line in markdown.splitlines():
            line = raw_line.strip()
            if line.startswith("### "):
                current_category = cls._clean_cell(line[4:])
                continue

            if not current_category or not line.startswith("|"):
                continue
            if ":---" in line or line.lower().startswith("| api |"):
                continue

            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            if len(cells) < 5:
                continue

            name, url = cls._parse_link_cell(cells[0])
            if not name or not url:
                continue

            entries.append(
                {
                    "name": name,
                    "url": url,
                    "description": cls._clean_cell(cells[1]),
                    "auth": cls._clean_cell(cells[2]),
                    "https": cls._clean_cell(cells[3]).lower() == "yes",
                    "cors": cls._clean_cell(cells[4]),
                    "category": current_category,
                }
            )

        return entries

    @staticmethod
    def _parse_link_cell(cell: str) -> tuple[str, str]:
        """Extrait le nom et l'URL d'une cellule Markdown.

        Args:
            cell (str): Cellule contenant un lien Markdown.

        Returns:
            tuple[str, str]: Nom et URL extraits.
        """
        match = re.search(r"\[(?P<name>.+?)\]\((?P<url>.+?)\)", cell)
        if not match:
            return "", ""
        return (
            PublicApiCatalogService._clean_cell(match.group("name")),
            match.group("url").strip(),
        )

    @staticmethod
    def _clean_cell(text: str) -> str:
        """Nettoie une cellule Markdown simple.

        Args:
            text (str): Texte brut a normaliser.

        Returns:
            str: Texte simplifie.
        """
        return re.sub(r"\s+", " ", text.replace("`", "").strip())

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Decoupe un texte libre en mots significatifs.

        Args:
            text (str): Texte libre a tokenizer.

        Returns:
            list[str]: Mots utiles pour le scoring.
        """
        return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) >= 3]
