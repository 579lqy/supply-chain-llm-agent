# -*- coding: utf-8 -*-
"""
关键词版确定性路由（LEGACY / 兜底实现）· 仅作为对照与离线兜底使用。

这是"上一个版本"的完整逻辑，原封不动保留在这里：
  - route()        关键词表命中决定 rag/tool/hybrid 三路
  - LocalRetriever TF-IDF 关键词检索
  - dispatch_tool() 正则抽参 + if/elif 派发
  - build_*_answer() 填空模板答案

它在新版里的两个角色：
  1) 当 LLM_BACKEND=rulebased（或没有 API key）时，LLM 版会调用它，保证 demo 零依赖可跑；
  2) 作为"关键词版 vs LLM 版"的对照基线，README 与评测直接对比两者差异。

注意：这一层**没有任何真正的 LLM 调用**——所有"智能"都在上面的规则里。
真正调用 LLM 的地方在 llm_backend.OpenAIBackend。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from mcp_tools import (
    abc_classification,
    avg_lead_time_by_location,
    find_sku,
    lead_time_stats,
    load_knowledge_documents,
    low_stock_skus,
    supplier_defect_ranking,
    supplier_scorecard,
)


# ---------------------------------------------------------------------------
# 1) 关键词意图路由（确定性）
# ---------------------------------------------------------------------------
FORMULA_TERMS = [
    "安全库存", "再订货点", "rop", "eoq", "经济订货批量", "abc", "aql",
    "评分卡", "公式", "服务水平", "经济订货",
]
CONCRETE_DATA_TERMS = [
    "sku0", "sku1", "sku2", "sku3", "sku4", "sku5",
    "mumbai", "kolkata", "chennai", "bangalore", "delhi",
    "库存水平最低", "最低", "最高", "低于", "平均", "各城市", "均值", "全部 sku",
]
SOP_TERMS = ["风险", "sop", "管控", "预警", "策略", "达标", "是否触发", "判为", "应采用"]


def route(query: str) -> str:
    q = query.lower()
    has_f = any(t in q for t in FORMULA_TERMS)
    has_cd = any(t in q for t in CONCRETE_DATA_TERMS)
    has_s = any(t in q for t in SOP_TERMS)
    if has_f and has_cd:
        return "hybrid"
    if has_s and has_cd:
        return "hybrid"
    if has_f:
        return "rag"
    if has_cd:
        return "tool"
    return "rag"


# ---------------------------------------------------------------------------
# 2) TF-IDF 关键词检索（确定性）
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9-]+|[\u4e00-\u9fff]", text.lower())
    stopwords = {"的", "和", "及", "与", "是", "了", "在", "到", "如何"}
    return [word for word in words if word not in stopwords]


class LocalRetriever:
    QUERY_SYNONYMS = {
        "补货": ["补货", "库存", "风险", "缺陷率"],
        "优先": ["优先", "评分卡", "风险", "管控"],
        "风险": ["风险", "缺陷率", "达标", "管控"],
    }

    def __init__(self, documents: list[dict[str, str]]) -> None:
        self.documents = documents
        self.doc_tokens = [tokenize(doc["content"]) for doc in documents]
        self.doc_freq = Counter()
        for tokens in self.doc_tokens:
            self.doc_freq.update(set(tokens))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        query_tokens = tokenize(query)
        expanded = list(query_tokens)
        for t in query_tokens:
            if t in self.QUERY_SYNONYMS:
                expanded.extend(self.QUERY_SYNONYMS[t])
        query_tokens = expanded
        total_docs = max(len(self.documents), 1)
        scored: list[tuple[float, dict[str, str]]] = []
        for doc, tokens in zip(self.documents, self.doc_tokens):
            token_counts = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                if token not in token_counts:
                    continue
                idf = math.log((total_docs + 1) / (self.doc_freq[token] + 1)) + 1
                score += token_counts[token] * idf
            if score > 0:
                scored.append((round(score, 4), doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "score": s, "id": doc["id"], "source": doc["source"],
                "title": doc["title"], "section_path": doc.get("section_path", doc["title"]),
                "content": doc["content"],
            }
            for s, doc in scored[:top_k]
        ]


# ---------------------------------------------------------------------------
# 3) 正则抽参 + 关键词派发（确定性）
# ---------------------------------------------------------------------------
def _extract_sku(query: str) -> str | None:
    m = re.search(r"sku\d+", query, re.IGNORECASE)
    return m.group(0).upper() if m else None


def _extract_location(query: str) -> str | None:
    for loc in ["Mumbai", "Kolkata", "Chennai", "Bangalore", "Delhi"]:
        if loc.lower() in query.lower():
            return loc
    return None


def _extract_threshold(query: str) -> int | None:
    m = re.search(r"低于\s*(\d+)", query)
    return int(m.group(1)) if m else None


def dispatch_tool(query: str) -> dict[str, Any]:
    q = query.lower()
    sku = _extract_sku(query)
    loc = _extract_location(query)
    thr = _extract_threshold(query)

    if "库存" in q:
        if thr is not None:
            return {"tool": "low_stock_skus", "result": low_stock_skus(location=loc, threshold=thr)}
        if "最低" in q:
            m = re.search(r"最低\D*(\d+)", q)
            top_n = int(m.group(1)) if m else 3
            return {"tool": "low_stock_skus", "result": low_stock_skus(location=loc, top_n=top_n)}
        return {"tool": "low_stock_skus", "result": low_stock_skus(location=loc)}
    if "缺陷率最高" in q or "平均缺陷率" in q:
        return {"tool": "supplier_defect_ranking", "result": supplier_defect_ranking()}
    if "平均提前期" in q or "各城市" in q:
        return {"tool": "avg_lead_time_by_location", "result": avg_lead_time_by_location()}
    if "均值" in q and "2σ" in q:
        return {"tool": "lead_time_stats", "result": lead_time_stats()}
    if "abc" in q or "营收" in q:
        return {"tool": "abc_classification", "result": abc_classification()}
    if "评分卡" in q:
        return {"tool": "supplier_scorecard", "result": supplier_scorecard(sku) if sku else {}}
    if sku and ("单价" in q or "运费" in q or "成本" in q or "缺陷率" in q or "提前期" in q):
        f = find_sku(sku)
        return {"tool": "find_sku", "result": _asdict(f) if f else {}}
    if sku:
        f = find_sku(sku)
        return {"tool": "find_sku", "result": _asdict(f) if f else {}}
    if "补货" in q and ("优先" in q or "先" in q):
        res = low_stock_skus(location=loc, top_n=5)
        if res:
            res.sort(key=lambda r: (r["stock_level"], -r["defect_rate"]))
            top = res[0]
            return {"tool": "priority_restock",
                    "result": f"Mumbai 最优先补货: {top['sku']}(库存{top['stock_level']},缺陷率{top['defect_rate']:.4f})"}
        return {"tool": None, "result": {}}
    return {"tool": None, "result": {}}


def _asdict(facts) -> dict[str, Any]:
    from dataclasses import asdict
    return asdict(facts)


# ---------------------------------------------------------------------------
# 4) 模板答案（确定性填空）
# ---------------------------------------------------------------------------
def build_rag_answer(query: str, contexts: list[dict[str, Any]]) -> dict[str, Any]:
    top = contexts[0] if contexts else {}
    return {
        "query": query,
        "scenario": "供应链知识 / 方法论问答（RAG 检索）",
        "findings": [f"依据「{top.get('section_path', '')}」：{top.get('content', '')[:240]}…"
                     if top else "未检索到相关知识。"],
        "strategy": ["直接套用上述 SOP/公式定义回答。"],
        "risks": ["需确认适用前提（如服务水平、持有成本率）与数据口径一致。"],
    }


def build_tool_answer(query: str, tool_out: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": query,
        "scenario": "供应链实时数值 / 统计问答（MCP 工具）",
        "findings": [f"调用工具 {tool_out['tool']}，结果：{_summarize(tool_out['result'])}"],
        "strategy": ["基于真实数据给出结论，无需文本检索。"],
        "risks": ["数值随数据集更新而变化，结论仅对当前快照有效。"],
    }


def build_hybrid_answer(query: str, contexts: list[dict[str, Any]],
                        tool_out: dict[str, Any]) -> dict[str, Any]:
    top = contexts[0] if contexts else {}
    return {
        "query": query,
        "scenario": "方法论 + 真实数据混合问答（RAG 公式 + 工具数值）",
        "findings": [
            f"公式/规则（RAG）：「{top.get('section_path', '')}」",
            f"数据（工具 {tool_out['tool']}）：{_summarize(tool_out['result'])}",
        ],
        "strategy": ["将真实数据代入 SOP 公式得到可执行结论。"],
        "risks": ["公式参数口径（如 σ_d、持有率）需与业务确认，避免误用默认假设。"],
    }


def _summarize(result: Any, limit: int = 3) -> str:
    if isinstance(result, list):
        return str(result[:limit]) + (f" …(共 {len(result)} 项)" if len(result) > limit else "")
    if isinstance(result, dict) and "ordered" in result:
        return f"top20%SKU 营收占比 {result['top20_revenue_share']}%，共 {len(result['ordered'])} 个 SKU"
    return str(result)[:240]


def keyword_plan(query: str) -> dict[str, Any]:
    """把关键词链路封装成与 LLM 后端一致的 plan 结构（供 RuleBasedBackend 复用）。"""
    intent = route(query)
    tool_out = dispatch_tool(query)
    tool_calls: list[dict[str, Any]] = []
    if intent in ("tool", "hybrid") and tool_out.get("tool"):
        tool_calls.append({"name": tool_out["tool"], "arguments": {}, "result": tool_out["result"]})
    return {"intent": intent, "needs_knowledge": intent in ("rag", "hybrid"), "tool_calls": tool_calls}
