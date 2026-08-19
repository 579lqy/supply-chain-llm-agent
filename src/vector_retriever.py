# -*- coding: utf-8 -*-
"""
向量检索（RAG 质量升级点）· 离线可跑，无 key 也能用。

与关键词版 LocalRetriever 的区别：
  - 关键词版：字符/TF-IDF 命中，靠同义词表打补丁，对"换种说法"的问法召回不稳。
  - 本模块：用 embedding 模型把"问题"和"知识块"映射到同一语义空间，按余弦相似度召回，
    真正理解语义（例如"该先补哪个货"能命中"再订货点/SOP"而非只命中"补货"二字）。

降级策略（保证 demo 永远可跑）：
  1) 优先 sentence-transformers 本地模型（默认 BAAI/bge-small-zh-v1.5，中文效果好、512 维、可离线）。
  2) 若未安装 sentence-transformers / 首次下载失败 → 自动回退到关键词版 LocalRetriever（TF-IDF）。
  3) 若设置了 OPENAI_EMBEDDING_MODEL 且有 key，也可走在线 embedding（可选，不默认）。

`search(query, top_k)` 接口与 LocalRetriever 完全一致，rag_agent 一行不用改。
"""
from __future__ import annotations

import math
from typing import Any

from mcp_tools import load_knowledge_documents


def _cosine(a: list[float], b: list[float]) -> float:
    """纯 Python 余弦相似度，避免强依赖 numpy（向量规模小，足够快）。"""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class VectorRetriever:
    def __init__(self, documents: list[dict[str, str]] | None = None,
                 model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        self.documents = documents if documents is not None else load_knowledge_documents()
        self.model_name = model_name
        self.mode = "none"
        self.model = None
        self.emb: list[list[float]] = []
        self.fallback = None
        self._try_init()

    def _try_init(self) -> None:
        # 1) 本地 sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            corpus = [d["content"] for d in self.documents]
            self.emb = [list(map(float, v)) for v in self.model.encode(corpus, normalize_embeddings=True)]
            self.mode = "vector"
            return
        except Exception:
            pass
        # 2) 在线 embedding（可选）
        import os
        if os.environ.get("OPENAI_EMBEDDING_MODEL") and os.environ.get("OPENAI_API_KEY"):
            try:
                self.emb = self._online_embed([d["content"] for d in self.documents])
                self.mode = "online"
                return
            except Exception:
                pass
        # 3) 回退 TF-IDF
        from keyword_router import LocalRetriever
        self.fallback = LocalRetriever(self.documents)
        self.mode = "tfidf"

    def _online_embed(self, texts: list[str]) -> list[list[float]]:
        import os
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                        base_url=os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1")
        out: list[list[float]] = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            resp = client.embeddings.create(model=os.environ["OPENAI_EMBEDDING_MODEL"], input=batch)
            out.extend([list(map(float, d.embedding)) for d in resp.data])
        return out

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self.mode == "tfidf":
            return self.fallback.search(query, top_k=top_k)
        qv = list(map(float, self.model.encode([query], normalize_embeddings=True)[0])) if self.mode == "vector" \
            else self._online_embed([query])[0]
        scored = [
            (_cosine(qv, doc_emb), doc)
            for doc, doc_emb in zip(self.documents, self.emb)
        ]
        # 与 LocalRetriever 输出字段对齐
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "score": round(s, 4), "id": doc["id"], "source": doc["source"],
                "title": doc["title"], "section_path": doc.get("section_path", doc["title"]),
                "content": doc["content"],
            }
            for s, doc in scored[:top_k]
        ]


def get_retriever() -> VectorRetriever:
    return VectorRetriever()
