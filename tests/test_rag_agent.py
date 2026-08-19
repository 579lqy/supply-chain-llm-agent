# -*- coding: utf-8 -*-
"""LLM 版 Agent 冒烟测试（零依赖，默认 rulebased 后端，不依赖 LLM）。

断言对象为「意图识别 + 场景识别 + 证据召回 / 工具调用」这一可复现行为。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_agent import LlmRagAgent  # noqa: E402


def _agent() -> LlmRagAgent:
    return LlmRagAgent()


def test_rag_intent() -> None:
    answer = _agent().run("如何计算安全库存？给出公式与各参数含义。")
    assert answer["intent"] == "rag"
    assert "RAG" in answer["scenario"] or "检索" in answer["scenario"]
    assert answer["evidence"], "RAG 类必须有召回证据"


def test_tool_intent() -> None:
    answer = _agent().run("各城市的平均提前期(lead_time_days)各是多少？")
    assert answer["intent"] in ("tool", "hybrid")
    assert answer["tool_calls"], "tool 类必须调用数据工具"
    assert answer["findings"], "tool 类必须给出数据结论"


def test_hybrid_intent() -> None:
    answer = _agent().run("Mumbai 仓哪个 SKU 最该优先补货（结合库存覆盖与缺陷风险）？")
    assert answer["intent"] == "hybrid"
    assert answer["evidence"], "hybrid 类必须同时有知识证据"
    assert answer["tool_calls"], "hybrid 类必须调用数据工具"
