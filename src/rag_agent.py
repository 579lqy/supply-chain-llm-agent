# -*- coding: utf-8 -*-
"""
LLM 版统一编排器：把"路由/检索/工具执行/答案合成"串成一条链路。

与关键词版的差异（这是整个升级的核心）：
  关键词版：route()(关键词) → LocalRetriever()(TF-IDF) → dispatch_tool()(正则) → 模板填空
  LLM 版  ：backend.plan()(LLM 理解/选工具) → VectorRetriever()(语义召回)
            → execute_tool()(确定性执行真实数据) → backend.synthesize()(LLM 生成)

确定性"手"——检索执行、真实数据计算、参数校验、输出结构——仍由代码负责；
LLM 只负责"脑"——理解问法、选工具、组织自然语言答案。这就是"RAG 管知识、MCP 管数据、LLM 管脑"的落地。
"""
from __future__ import annotations

from typing import Any

from llm_backend import get_backend
from mcp_tools import tool_manifest
from vector_retriever import get_retriever


def execute_tool(name: str, arguments: dict[str, Any]) -> Any:
    """按 LLM 选出的工具名执行确定性真实数据工具（绝不让 LLM 直接算数）。"""
    import inspect
    from dataclasses import asdict, is_dataclass

    import mcp_tools
    fn = getattr(mcp_tools, name, None)
    if fn is None:
        return {"error": f"未知工具: {name}"}
    # 只传工具声明的参数，避免 LLM 瞎给字段导致 TypeError
    sig = inspect.signature(fn)
    kwargs = {k: v for k, v in (arguments or {}).items() if k in sig.parameters}
    try:
        res = fn(**kwargs)
    except Exception as e:  # 工具执行失败也要有返回，不让链路崩
        return {"error": f"{name} 执行失败: {e}"}
    # 统一规整为 dict（find_sku 返回 dataclass，便于下游评测/忠实度检查）
    if is_dataclass(res):
        return asdict(res)
    if isinstance(res, list) and res and is_dataclass(res[0]):
        return [asdict(x) for x in res]
    return res


class LlmRagAgent:
    def __init__(self, backend=None, retriever=None) -> None:
        self.backend = backend or get_backend()
        self.retriever = retriever or get_retriever()
        self.manifest = tool_manifest()["tools"]

    def run(self, query: str, top_k: int = 5) -> dict[str, Any]:
        # ① LLM（或规则兜底）理解问法、决定调哪些工具
        plan = self.backend.plan(query, self.manifest)
        # ② 语义检索知识库（RAG）
        contexts = self.retriever.search(query, top_k=top_k) if plan.get("needs_knowledge") else []
        # ③ 确定性执行真实数据工具（LLM 后端返回 result=None，这里补执行；规则后端已预填）
        tool_results: list[dict[str, Any]] = []
        for tc in plan.get("tool_calls", []):
            if tc.get("result") is None:
                tc["result"] = execute_tool(tc["name"], tc.get("arguments", {}))
            tool_results.append({
                "tool": tc["name"],
                "arguments": tc.get("arguments", {}),
                "result": tc["result"],
            })
        # ④ LLM（或规则兜底）把知识与数据综合成结构化答案
        answer = self.backend.synthesize(query, contexts, tool_results, plan)
        # ⑤ 统一输出契约（前端零改动）
        answer["query"] = query
        answer["intent"] = plan["intent"]
        answer["backend"] = self.backend.name
        answer["retriever_mode"] = getattr(self.retriever, "mode", "unknown")
        answer["evidence"] = [
            {"source": c["source"], "section_path": c["section_path"], "score": c["score"]}
            for c in contexts
        ]
        answer["tool_calls"] = tool_results
        answer["retrieved_context"] = contexts
        return answer


def format_answer(answer: dict[str, Any]) -> str:
    lines = [
        "# RAG Agent 策略输出",
        "",
        f"输入需求：{answer['query']}",
        f"后端：{answer.get('backend', '?')} ｜ 检索：{answer.get('retriever_mode', '?')} ｜ "
        f"意图：{answer['intent']} ｜ 场景：{answer.get('scenario', '')}",
        "",
        "## 关键发现",
    ]
    lines.extend(f"- {item}" for item in answer.get("findings", []))
    lines.append("")
    lines.append("## 推荐策略")
    lines.extend(f"- {item}" for item in answer.get("strategy", []))
    lines.append("")
    lines.append("## 风险提示")
    lines.extend(f"- {item}" for item in answer.get("risks", []))
    if answer.get("evidence"):
        lines.append("")
        lines.append("## 依据来源（RAG 召回）")
        lines.extend(
            f"- {item['source']} / {item['section_path']}，相关度 {item['score']}"
            for item in answer["evidence"]
        )
    if answer.get("tool_calls"):
        lines.append("")
        lines.append("## 调用的真实数据工具")
        for tc in answer["tool_calls"]:
            lines.append(f"- {tc['tool']}({tc.get('arguments', {})})")
    return "\n".join(lines)
