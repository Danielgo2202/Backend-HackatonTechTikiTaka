"""Fetch active CRM client row from Supabase."""

from __future__ import annotations

import logging
from typing import Any

from supabase import Client, create_client

from config import Settings, get_settings

logger = logging.getLogger(__name__)


def _client(settings: Settings) -> Client | None:
    if not settings.supabase_url or not settings.supabase_key:
        return None
    return create_client(settings.supabase_url, settings.supabase_key)


def fetch_active_client_context(settings: Settings | None = None) -> dict[str, Any] | None:
    settings = settings or get_settings()
    sb = _client(settings)
    if sb is None:
        return None
    try:
        res = sb.table("clients").select("*").eq("active", True).limit(1).execute()
        rows = res.data or []
        if not rows:
            return None
        row = rows[0]
        return {
            "name": row.get("name"),
            "industry": row.get("industry"),
            "deal_size": row.get("deal_size"),
            "pain_points": row.get("pain_points"),
        }
    except Exception as e:
        logger.warning("Supabase fetch failed (demo can continue without CRM): %s", e)
        return None
