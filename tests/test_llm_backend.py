# -*- coding: utf-8 -*-
"""LLM 后端与检索层单测（零依赖，不调用真实 LLM / 不下载模型）。

覆盖：规则后端与关键词版一致、function-calling schema 转换、JSON 解析鲁棒性、
向量检索降级、工具执行、Agent 端到端 run。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_backend import (  # noqa: E402
    RuleBasedBackend,
    _extract_json,
    coerce_args,
    manifest_to_openai_tools,
)
from mcp_tools import tool_manifest  # noqa: E402
from rag_agent import LlmRagAgent, execute_tool  # noqa: E402
from vector_retriever import get_retriever  # noqa: E402


def test_rulebased_plan_matches_keyword() -> None:
    from keyword_router import route
    for q in ["如何计算安全库存？", "各城市的平均提前期各是多少？",
              "Mumbai 仓哪个 SKU 最该优先补货？"]:
        plan = RuleBasedBackend().plan(q, tool_manifest()["tools"])
        assert plan["intent"] == route(q), q


def test_manifest_to_openai_tools() -> None:
    tools = manifest_to_openai_tools(tool_manifest()["tools"])
    assert tools[0]["type"] == "function"
    fn = {t["function"]["name"]: t["function"] for t in tools}
    assert "find_sku" in fn
    assert fn["find_sku"]["parameters"]["properties"]["sku"]["type"] == "string"
    # 可选参数（带 ?）不应出现在 required
    loc = fn["low_stock_skus"]["parameters"]
    assert "location" not in loc["required"]


def test_coerce_args() -> None:
    assert coerce_args({"top_n": "5"}) == {"top_n": 5}
    assert coerce_args({"location": "Mumbai"}) == {"location": "Mumbai"}


def test_extract_json_robust() -> None:
    assert _extract_json('{"a":1}') == {"a": 1}
    assert _extract_json("```json\n{\"a\":1}\n```") == {"a": 1}
    assert _extract_json("前缀 {\"a\":1} 后缀") == {"a": 1}
    assert _extract_json("not json") == {}


def test_vector_retriever_fallback() -> None:
    retr = get_retriever()
    assert retr.mode in ("vector", "online", "tfidf")
    out = retr.search("安全库存怎么算", top_k=3)
    assert isinstance(out, list) and len(out) <= 3
    for c in out:
        assert {"score", "source", "section_path", "content"} <= set(c)


def test_execute_tool() -> None:
    res = execute_tool("find_sku", {"sku": "SKU0"})
    assert isinstance(res, dict) and res.get("sku", "").upper() == "SKU0"


def test_agent_run_rulebased() -> None:
    ans = LlmRagAgent().run("SKU0 的缺陷率是否超过 AQL 2.5%？")
    assert ans["backend"] == "rulebased"
    assert ans["tool_calls"]
    assert ans["findings"]
