"""Construit des contextes consultatifs prop et investissement."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

import yfinance as yf
from shared import InvestmentThesisEnvelope, MarketContextEnvelope, get_settings
from shared.redis_client import get_redis_client

from eva_researcher.services.search import ResearchQuery, ResearchService

logger = logging.getLogger(__name__)


POSITIVE_KEYWORDS = {"beat", "growth", "upgrade", "bullish", "strong", "record", "expand", "tailwind"}
NEGATIVE_KEYWORDS = {"downgrade", "miss", "bearish", "weak", "lawsuit", "fraud", "warning", "outflow"}
EVENT_KEYWORDS = {"cpi", "fomc", "nfp", "earnings", "rate", "inflation", "ecb", "fed", "banque centrale"}
GEO_KEYWORDS = {"war", "sanction", "tariff", "conflict", "embargo", "election", "geopolitic", "geopolitical"}


class ContextEngine:
    """Construit, met en cache et publie les contextes marches et theses."""

    def __init__(self, research_service: ResearchService) -> None:
        self.research_service = research_service
        self.settings = get_settings()
        self._market_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._invest_cache: dict[str, dict[str, Any]] = {}

    async def build_market_context(
        self,
        *,
        symbol: str,
        family: str = "mixed",
        ttl_seconds: int = 900,
    ) -> dict[str, Any]:
        """Construit un contexte prop consultatif pour un symbole."""

        normalized_symbol = str(symbol).strip().upper()
        query = f"{normalized_symbol} macro geopolitique news market risk"
        search_payload = await self.research_service.search(
            ResearchQuery(
                query=query,
                domain="finance",
                depth="quick",
                max_results=6,
            )
        )
        results = list(search_payload.get("results") or [])
        text_blob = " ".join(
            f"{item.get('title', '')} {item.get('summary', '')}".lower()
            for item in results
        )
        positive_hits = sum(1 for token in POSITIVE_KEYWORDS if token in text_blob)
        negative_hits = sum(1 for token in NEGATIVE_KEYWORDS if token in text_blob)
        event_hits = sum(1 for token in EVENT_KEYWORDS if token in text_blob)
        geo_hits = sum(1 for token in GEO_KEYWORDS if token in text_blob)

        if positive_hits > negative_hits:
            macro_bias = "bullish"
        elif negative_hits > positive_hits:
            macro_bias = "bearish"
        else:
            macro_bias = "neutral"

        if event_hits >= 3:
            event_risk = "high"
        elif event_hits >= 1:
            event_risk = "medium"
        else:
            event_risk = "low"

        if geo_hits >= 2:
            geo_risk = "high"
        elif geo_hits == 1:
            geo_risk = "medium"
        else:
            geo_risk = "low"

        blocked = event_risk == "high" or geo_risk == "high"
        confidence = min(0.95, 0.35 + (0.08 * len(results)) + (0.04 * (positive_hits + negative_hits + event_hits)))
        sources = [str(item.get("source") or "web") for item in results[:6]]
        generated_at = datetime.now()
        expires_at = generated_at + timedelta(seconds=max(ttl_seconds, 60))

        envelope = MarketContextEnvelope(
            symbol=normalized_symbol,
            family=str(family or "mixed").strip().lower() or "mixed",
            macro_bias=macro_bias,
            event_risk=event_risk,
            geo_risk=geo_risk,
            blocked=blocked,
            confidence=round(confidence, 3),
            sources=sources,
            payload={
                "query": query,
                "results": results,
                "synthesis": search_payload.get("synthesis"),
                "signals": {
                    "positive_hits": positive_hits,
                    "negative_hits": negative_hits,
                    "event_hits": event_hits,
                    "geo_hits": geo_hits,
                },
            },
            metadata={"ttl_seconds": ttl_seconds},
            generated_at=generated_at,
            ttl_seconds=int(ttl_seconds),
        )
        payload = envelope.model_dump()
        payload["snapshot_id"] = str(envelope.envelope_id)
        payload["generated_at"] = generated_at.isoformat()
        payload["expires_at"] = expires_at.isoformat()

        cache_key = self._market_cache_key(normalized_symbol, family)
        self._market_cache[(normalized_symbol, str(family or "mixed").lower())] = payload
        await self._publish_market_context(cache_key, payload, ttl_seconds)
        self._record_market_context_snapshot(payload)
        return payload

    async def get_latest_market_context(self, *, symbol: str, family: str = "mixed") -> dict[str, Any] | None:
        """Retourne le dernier contexte prop connu pour un symbole."""

        normalized_symbol = str(symbol).strip().upper()
        normalized_family = str(family or "mixed").strip().lower() or "mixed"
        local_payload = self._market_cache.get((normalized_symbol, normalized_family))
        if local_payload:
            return local_payload

        try:
            redis = get_redis_client()
            payload = await redis.cache_get(self._market_cache_key(normalized_symbol, normalized_family))
            return payload if isinstance(payload, dict) else None
        except Exception as exc:
            logger.debug("Lecture Redis du contexte marche ignoree: %s", exc)
            return None

    async def build_investment_thesis(
        self,
        *,
        symbol: str,
        issuer: str | None = None,
        horizon_months: int = 12,
    ) -> dict[str, Any]:
        """Construit une these investissement consultative longue."""

        normalized_symbol = str(symbol).strip().upper()
        ticker = yf.Ticker(normalized_symbol)
        info = ticker.info or {}
        history = ticker.history(period="6mo")
        current_price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0.0)
        target_price = float(info.get("targetMeanPrice") or 0.0)
        revenue_growth = float(info.get("revenueGrowth") or 0.0)
        operating_margin = float(info.get("operatingMargins") or 0.0)
        debt_to_equity = float(info.get("debtToEquity") or 0.0)

        conviction_score = 0.5
        if target_price > 0 and current_price > 0 and target_price > current_price:
            conviction_score += 0.15
        if revenue_growth > 0:
            conviction_score += 0.15
        if operating_margin > 0.1:
            conviction_score += 0.1
        if debt_to_equity > 180:
            conviction_score -= 0.15
        conviction_score = max(0.0, min(conviction_score, 0.95))

        if debt_to_equity > 220:
            fundamental_risk = "high_leverage"
        elif revenue_growth < 0:
            fundamental_risk = "weak_growth"
        else:
            fundamental_risk = "contained"

        governance_risk = "standard"
        if str(info.get("auditRisk") or "") not in {"", "None"}:
            try:
                audit_risk = int(info.get("auditRisk"))
                if audit_risk >= 7:
                    governance_risk = "high"
                elif audit_risk >= 4:
                    governance_risk = "medium"
            except Exception:
                governance_risk = "standard"

        trend_text = "historique indisponible"
        if not history.empty:
            first_close = float(history["Close"].iloc[0])
            last_close = float(history["Close"].iloc[-1])
            if first_close > 0:
                trend_pct = ((last_close - first_close) / first_close) * 100.0
                trend_text = f"variation 6 mois {trend_pct:.2f}%"

        thesis = (
            f"{issuer or info.get('shortName') or normalized_symbol}: conviction={conviction_score:.2f}, "
            f"croissance={revenue_growth:.2f}, marge_op={operating_margin:.2f}, "
            f"dette_fonds_propres={debt_to_equity:.2f}, {trend_text}."
        )

        envelope = InvestmentThesisEnvelope(
            symbol=normalized_symbol,
            issuer=str(issuer or info.get("shortName") or "").strip() or None,
            conviction_score=round(conviction_score, 3),
            fundamental_risk=fundamental_risk,
            governance_risk=governance_risk,
            horizon_months=max(int(horizon_months), 1),
            thesis=thesis,
            sources=["yfinance", "researcher"],
            payload={
                "info": {
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "target_mean_price": target_price,
                    "current_price": current_price,
                    "revenue_growth": revenue_growth,
                    "operating_margin": operating_margin,
                    "debt_to_equity": debt_to_equity,
                },
                "history_rows": int(len(history.index)),
            },
            generated_at=datetime.now(),
            review_status="draft",
        )
        payload = envelope.model_dump()
        payload["thesis_id"] = str(envelope.envelope_id)
        payload["generated_at"] = envelope.generated_at.isoformat()

        self._invest_cache[normalized_symbol] = payload
        await self._publish_investment_thesis(normalized_symbol, payload)
        self._record_investment_thesis(payload)
        return payload

    async def get_latest_investment_thesis(self, *, symbol: str) -> dict[str, Any] | None:
        """Retourne la derniere these investissement connue."""

        normalized_symbol = str(symbol).strip().upper()
        local_payload = self._invest_cache.get(normalized_symbol)
        if local_payload:
            return local_payload

        try:
            redis = get_redis_client()
            payload = await redis.cache_get(self._investment_cache_key(normalized_symbol))
            return payload if isinstance(payload, dict) else None
        except Exception as exc:
            logger.debug("Lecture Redis de la these investissement ignoree: %s", exc)
            return None

    def _market_cache_key(self, symbol: str, family: str) -> str:
        """Construit la cle Redis du contexte prop."""

        return f"eva:state:intelligence:market_context:{str(family).lower()}:{symbol.upper()}"

    def _investment_cache_key(self, symbol: str) -> str:
        """Construit la cle Redis de these investissement."""

        return f"eva:state:intelligence:investment_thesis:{symbol.upper()}"

    async def _publish_market_context(self, cache_key: str, payload: dict[str, Any], ttl_seconds: int) -> None:
        """Publie un contexte marche dans Redis."""

        try:
            redis = get_redis_client()
            await redis.cache_set(cache_key, payload, ttl_seconds=ttl_seconds)
            await redis.publish("eva.intelligence.market_context", payload)
        except Exception as exc:
            logger.debug("Publication Redis du contexte marche ignoree: %s", exc)

    async def _publish_investment_thesis(self, symbol: str, payload: dict[str, Any]) -> None:
        """Publie une these investissement dans Redis."""

        try:
            redis = get_redis_client()
            await redis.cache_set(self._investment_cache_key(symbol), payload, ttl_seconds=86400)
            await redis.publish("eva.intelligence.investment_thesis", payload)
        except Exception as exc:
            logger.debug("Publication Redis de la these investissement ignoree: %s", exc)

    def _record_market_context_snapshot(self, payload: dict[str, Any]) -> None:
        """Persiste un contexte marche dans TimeDB si disponible."""

        self._record_research_row(
            table_name="research.market_context_snapshots",
            columns=[
                "snapshot_id",
                "symbol",
                "family",
                "mode",
                "macro_bias",
                "event_risk",
                "geo_risk",
                "blocked",
                "confidence",
                "sources",
                "payload",
                "generated_at",
                "expires_at",
            ],
            values=[
                payload.get("snapshot_id"),
                payload.get("symbol"),
                payload.get("family"),
                payload.get("mode"),
                payload.get("macro_bias"),
                payload.get("event_risk"),
                payload.get("geo_risk"),
                payload.get("blocked"),
                payload.get("confidence"),
                json.dumps(payload.get("sources") or [], ensure_ascii=False, default=str),
                json.dumps(payload, ensure_ascii=False, default=str),
                payload.get("generated_at"),
                payload.get("expires_at"),
            ],
            conflict_key="snapshot_id",
        )

    def _record_investment_thesis(self, payload: dict[str, Any]) -> None:
        """Persiste une these investissement dans TimeDB si disponible."""

        self._record_research_row(
            table_name="research.investment_theses",
            columns=[
                "thesis_id",
                "symbol",
                "issuer",
                "mode",
                "conviction_score",
                "fundamental_risk",
                "governance_risk",
                "horizon_months",
                "thesis",
                "sources",
                "payload",
                "generated_at",
                "review_status",
            ],
            values=[
                payload.get("thesis_id"),
                payload.get("symbol"),
                payload.get("issuer"),
                payload.get("mode"),
                payload.get("conviction_score"),
                payload.get("fundamental_risk"),
                payload.get("governance_risk"),
                payload.get("horizon_months"),
                payload.get("thesis"),
                json.dumps(payload.get("sources") or [], ensure_ascii=False, default=str),
                json.dumps(payload, ensure_ascii=False, default=str),
                payload.get("generated_at"),
                payload.get("review_status"),
            ],
            conflict_key="thesis_id",
        )

    def _record_research_row(
        self,
        *,
        table_name: str,
        columns: list[str],
        values: list[Any],
        conflict_key: str,
    ) -> None:
        """Insere une ligne dans TimeDB pour la recherche si la base est active."""

        if not self._timescale_enabled():
            return

        try:
            import psycopg2
        except Exception:
            logger.debug("psycopg2 indisponible pour la persistence researcher.")
            return

        placeholders = ", ".join(["%s"] * len(values))
        update_parts = [f"{column} = EXCLUDED.{column}" for column in columns if column != conflict_key]
        query = (
            f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders}) "
            f"ON CONFLICT ({conflict_key}) DO UPDATE SET {', '.join(update_parts)}"
        )

        try:
            with psycopg2.connect(self._timescale_dsn()) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("CREATE SCHEMA IF NOT EXISTS research")
                    if table_name.endswith("market_context_snapshots"):
                        cursor.execute(
                            """
                            CREATE TABLE IF NOT EXISTS research.market_context_snapshots (
                                snapshot_id TEXT PRIMARY KEY,
                                symbol TEXT NOT NULL,
                                family TEXT NULL,
                                mode TEXT NOT NULL DEFAULT 'prop',
                                macro_bias TEXT NULL,
                                event_risk TEXT NULL,
                                geo_risk TEXT NULL,
                                blocked BOOLEAN NOT NULL DEFAULT FALSE,
                                confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                                sources JSONB NOT NULL DEFAULT '[]'::jsonb,
                                payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                                generated_at TIMESTAMPTZ NOT NULL,
                                expires_at TIMESTAMPTZ NULL
                            )
                            """
                        )
                    elif table_name.endswith("investment_theses"):
                        cursor.execute(
                            """
                            CREATE TABLE IF NOT EXISTS research.investment_theses (
                                thesis_id TEXT PRIMARY KEY,
                                symbol TEXT NOT NULL,
                                issuer TEXT NULL,
                                mode TEXT NOT NULL DEFAULT 'invest',
                                conviction_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                                fundamental_risk TEXT NULL,
                                governance_risk TEXT NULL,
                                horizon_months INTEGER NOT NULL DEFAULT 12,
                                thesis TEXT NOT NULL DEFAULT '',
                                sources JSONB NOT NULL DEFAULT '[]'::jsonb,
                                payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                                generated_at TIMESTAMPTZ NOT NULL,
                                review_status TEXT NOT NULL DEFAULT 'draft'
                            )
                            """
                        )
                    cursor.execute(query, tuple(values))
                connection.commit()
        except Exception as exc:
            logger.debug("Persistence TimeDB researcher ignoree: %s", exc)

    def _timescale_enabled(self) -> bool:
        """Retourne l'etat de la persistence consultative TimeDB."""

        raw_value = os.getenv("TRAINING_TIMESCALE_ENABLED", "0").strip().lower()
        return raw_value in {"1", "true", "yes", "on"}

    def _timescale_dsn(self) -> str:
        """Construit le DSN PostgreSQL/TimescaleDB du service researcher."""

        return (
            f"host={os.getenv('TRAINING_TIMESCALE_HOST', os.getenv('TIMESCALE_HOST', 'timescaledb'))} "
            f"port={os.getenv('TRAINING_TIMESCALE_PORT', os.getenv('TIMESCALE_PORT', '5432'))} "
            f"dbname={os.getenv('TRAINING_TIMESCALE_DB', os.getenv('TIMESCALE_DB', 'thehive'))} "
            f"user={os.getenv('TRAINING_TIMESCALE_USER', os.getenv('TIMESCALE_USER', 'eva'))} "
            f"password={os.getenv('TRAINING_TIMESCALE_PASSWORD', os.getenv('TIMESCALE_PASSWORD', ''))}"
        )
