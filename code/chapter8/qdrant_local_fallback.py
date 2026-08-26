#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 8 章记忆后端兜底。

当前环境里的 Qdrant Cloud 对 REST 返回 404，Neo4j Aura 域名无法解析；
本机也没有 Docker。演示脚本在创建 MemoryTool 前调用 ensure_qdrant()：

- Qdrant 不可用 → qdrant-client 本地 path 模式（无需独立服务）
- Neo4j 不可用 → 进程内图存储，接口与 Neo4jGraphStore 对齐
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_CHAPTER_DIR = Path(__file__).resolve().parent
_ENV_PATH = _CHAPTER_DIR / ".env"
_LOCAL_PATH = _CHAPTER_DIR / "memory_data" / "qdrant_local"
_PATCHED = False
_NEO4J_PATCHED = False
_LOCAL_CLIENT = None


def ensure_qdrant() -> None:
    """在创建 MemoryTool 之前调用。云端可用则保留；否则切到本地后端。"""
    global _PATCHED
    if _PATCHED:
        return

    load_dotenv(_ENV_PATH)
    url = (os.getenv("QDRANT_URL") or "").strip()
    api_key = (os.getenv("QDRANT_API_KEY") or "").strip()

    if not (url and _cloud_is_reachable(url, api_key)):
        _patch_vector_store_to_local()

    if not _neo4j_is_reachable():
        _patch_neo4j_to_memory()

    _PATCHED = True


def _cloud_is_reachable(url: str, api_key: str) -> bool:
    from qdrant_client import QdrantClient
    from qdrant_client.http.exceptions import UnexpectedResponse

    # 官方云端 REST 在 :6333；缺端口时补上再试一次
    candidates = [url]
    if url.startswith("http") and not url.rstrip("/").endswith(":6333"):
        candidates.append(url.rstrip("/") + ":6333")

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            client = QdrantClient(
                url=candidate,
                api_key=api_key or None,
                timeout=15,
                port=None,
            )
            client.get_collections()
            if candidate != url:
                os.environ["QDRANT_URL"] = candidate
            print(f"✅ Qdrant 云服务可用: {candidate}")
            return True
        except UnexpectedResponse as exc:
            last_error = exc
        except Exception as exc:  # noqa: BLE001 - 探测失败即回退
            last_error = exc

    print(
        "⚠️ Qdrant 云服务不可用"
        + (f"（{last_error}）" if last_error else "")
        + "，改用本地嵌入式存储。"
    )
    print(f"   本地路径: {_LOCAL_PATH}")
    return False


def _patch_vector_store_to_local() -> None:
    """同一 path 只能有一个 QdrantClient，必须在多种记忆类型之间复用。"""
    global _LOCAL_CLIENT

    from hello_agents.memory.storage.qdrant_store import QdrantVectorStore
    from qdrant_client import QdrantClient

    _LOCAL_PATH.mkdir(parents=True, exist_ok=True)

    def _initialize_client(self) -> None:  # noqa: ANN001
        global _LOCAL_CLIENT
        try:
            if _LOCAL_CLIENT is None:
                _LOCAL_CLIENT = QdrantClient(path=str(_LOCAL_PATH))
            self.client = _LOCAL_CLIENT
            logger.info("✅ 使用本地嵌入式 Qdrant: %s", _LOCAL_PATH)
            self.client.get_collections()
            self._ensure_collection()
        except Exception as exc:
            logger.error("❌ 本地嵌入式 Qdrant 初始化失败: %s", exc)
            raise

    QdrantVectorStore._initialize_client = _initialize_client
    # 避免框架仍按云 URL 建连
    os.environ.pop("QDRANT_URL", None)
    os.environ.pop("QDRANT_API_KEY", None)


def _neo4j_is_reachable() -> bool:
    uri = (os.getenv("NEO4J_URI") or "").strip()
    user = (os.getenv("NEO4J_USERNAME") or "neo4j").strip()
    password = (os.getenv("NEO4J_PASSWORD") or "").strip()
    if not uri:
        print("⚠️ 未配置 NEO4J_URI，改用内存图存储。")
        return False
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(uri, auth=(user, password), connection_acquisition_timeout=8)
        try:
            driver.verify_connectivity()
        finally:
            driver.close()
        print(f"✅ Neo4j 可用: {uri}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ Neo4j 不可用（{exc}），改用内存图存储。")
        return False


class _InMemoryGraphStore:
    """与 Neo4jGraphStore 同接口的进程内图，仅供本地演示。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.entities: dict[str, dict[str, Any]] = {}
        self.relationships: list[dict[str, Any]] = []
        logger.info("✅ 使用内存图存储替代 Neo4j")

    def health_check(self) -> bool:
        return True

    def add_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        props = dict(properties or {})
        existing = self.entities.get(entity_id, {})
        props.update({"id": entity_id, "name": name, "type": entity_type})
        if existing:
            props["frequency"] = existing.get("frequency", 1) + 1
        else:
            props.setdefault("frequency", 1)
        self.entities[entity_id] = props
        return True

    def add_relationship(
        self,
        from_entity_id: str,
        to_entity_id: str,
        relationship_type: str,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        self.relationships.append(
            {
                "from": from_entity_id,
                "to": to_entity_id,
                "type": relationship_type,
                "properties": dict(properties or {}),
            }
        )
        return True

    def search_entities_by_name(
        self,
        name_pattern: str,
        entity_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        needle = (name_pattern or "").lower()
        hits = []
        for entity in self.entities.values():
            if needle and needle not in str(entity.get("name", "")).lower():
                continue
            if entity_types and entity.get("type") not in entity_types:
                continue
            hits.append(dict(entity))
            if len(hits) >= limit:
                break
        return hits

    def find_related_entities(
        self,
        entity_id: str,
        relationship_types: list[str] | None = None,
        max_depth: int = 2,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        seen = {entity_id}
        frontier = {entity_id}
        related: list[dict[str, Any]] = []
        for _ in range(max(1, max_depth)):
            nxt = set()
            for rel in self.relationships:
                if relationship_types and rel["type"] not in relationship_types:
                    continue
                neighbor = None
                if rel["from"] in frontier:
                    neighbor = rel["to"]
                elif rel["to"] in frontier:
                    neighbor = rel["from"]
                if neighbor and neighbor not in seen:
                    nxt.add(neighbor)
                    data = dict(self.entities.get(neighbor, {"id": neighbor}))
                    data.setdefault("id", neighbor)
                    data.update(rel.get("properties") or {})
                    related.append(data)
                    if len(related) >= limit:
                        return related
            frontier = nxt
            seen.update(nxt)
            if not frontier:
                break
        return related

    def get_entity_relationships(self, entity_id: str) -> list[dict[str, Any]]:
        result = []
        for rel in self.relationships:
            if rel["from"] == entity_id:
                result.append(
                    {
                        "relationship": {"type": rel["type"], **rel["properties"]},
                        "other": dict(self.entities.get(rel["to"], {"id": rel["to"]})),
                        "direction": "outgoing",
                    }
                )
            elif rel["to"] == entity_id:
                result.append(
                    {
                        "relationship": {"type": rel["type"], **rel["properties"]},
                        "other": dict(self.entities.get(rel["from"], {"id": rel["from"]})),
                        "direction": "incoming",
                    }
                )
        return result

    def delete_entity(self, entity_id: str) -> bool:
        existed = entity_id in self.entities
        self.entities.pop(entity_id, None)
        self.relationships = [
            rel
            for rel in self.relationships
            if rel["from"] != entity_id and rel["to"] != entity_id
        ]
        return existed

    def clear_all(self) -> bool:
        self.entities.clear()
        self.relationships.clear()
        return True

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_nodes": len(self.entities),
            "entity_nodes": len(self.entities),
            "memory_nodes": 0,
            "total_relationships": len(self.relationships),
        }


def _patch_neo4j_to_memory() -> None:
    global _NEO4J_PATCHED
    if _NEO4J_PATCHED:
        return
    from hello_agents.memory.storage import neo4j_store

    neo4j_store.Neo4jGraphStore = _InMemoryGraphStore
    os.environ.pop("NEO4J_URI", None)
    _NEO4J_PATCHED = True
