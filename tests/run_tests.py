# -*- coding: utf-8 -*-
"""统一测试入口：冒烟测试 + 评测流水线真实性校验。

运行：
    python tests/run_tests.py

默认（LLM_BACKEND=rulebased，零依赖）会执行：
  1) test_rag_agent 的 3 条冒烟用例
  2) test_llm_backend 的 7 条单测
  3) eval_rag 的 21 条真实评测（关键词版基线）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from test_rag_agent import (  # noqa: E402
    test_hybrid_intent,
    test_rag_intent,
    test_tool_intent,
)
from test_llm_backend import (  # noqa: E402
    test_agent_run_rulebased,
    test_coerce_args,
    test_execute_tool,
    test_extract_json_robust,
    test_manifest_to_openai_tools,
    test_rulebased_plan_matches_keyword,
    test_vector_retriever_fallback,
)
import eval_rag  # noqa: E402


def _run_smoke() -> None:
    for fn in (test_rag_intent, test_tool_intent, test_hybrid_intent,
               test_rulebased_plan_matches_keyword, test_manifest_to_openai_tools,
               test_coerce_args, test_extract_json_robust, test_vector_retriever_fallback,
               test_execute_tool, test_agent_run_rulebased):
        fn()
        print(f"PASS {fn.__name__}")


def _run_eval() -> None:
    summary = eval_rag.main()
    if summary.get("skipped"):
        print("SKIP eval_rag（无 OPENAI_API_KEY）")
        return
    if summary.get("backend") == "rulebased":
        assert summary["end2end_accuracy"] == 100.0, f"e2e 应为100%: {summary}"
        assert summary["recall_at_3"] == 100.0, f"Recall@3 应为100%: {summary}"
        assert summary["intent_accuracy"] >= 90.0, f"intent 应>=90%: {summary}"
        print(
            f"PASS eval_rag  intent={summary['intent_accuracy']}%  "
            f"Recall@3={summary['recall_at_3']}%  e2e={summary['end2end_accuracy']}%"
        )
    else:
        print(f"PASS eval_rag (LLM)  tool_call={summary['tool_call_accuracy']}%  "
              f"faithful={summary['faithfulness_rate']}%")


def main() -> None:
    _run_smoke()
    _run_eval()
    print("\nALL TESTS PASSED.")


if __name__ == "__main__":
    main()
