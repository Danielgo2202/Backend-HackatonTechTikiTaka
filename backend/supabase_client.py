"""Supabase helpers for active context and client search."""

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
        return _serialize_client_row(row)
    except Exception as e:
        logger.warning("Supabase fetch failed (demo can continue without CRM): %s", e)
        return None


def _serialize_client_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "name": row.get("name"),
        "industry": row.get("industry"),
        "deal_size": row.get("deal_size"),
        "pain_points": row.get("pain_points") or [],
        "active": bool(row.get("active", False)),
    }


def search_clients_by_name(
    query: str,
    limit: int = 5,
    settings: Settings | None = None,
    raise_on_error: bool = False,
) -> list[dict[str, Any]]:
    settings = settings or get_settings()
    sb = _client(settings)
    q = query.strip()
    if sb is None or not q:
        return []
    try:
        safe_limit = max(1, min(limit, 20))
        res = (
            sb.table("clients")
            .select("*")
            .ilike("name", f"%{q}%")
            .limit(safe_limit)
            .execute()
        )
        rows = res.data or []
        return [_serialize_client_row(row) for row in rows]
    except Exception as e:
        logger.warning("Supabase client search failed: %s", e)
        if raise_on_error:
            raise
        return []


def get_client_by_id(
    client_id: str,
    settings: Settings | None = None,
) -> dict[str, Any] | None:
    settings = settings or get_settings()
    sb = _client(settings)
    cid = client_id.strip()
    if sb is None or not cid:
        return None
    try:
        res = sb.table("clients").select("*").eq("id", cid).limit(1).execute()
        rows = res.data or []
        if not rows:
            return None
        return _serialize_client_row(rows[0])
    except Exception as e:
        logger.warning("Supabase get client by id failed: %s", e)
        return None
