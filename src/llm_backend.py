# -*- coding: utf-8 -*-
"""LLM 后端抽象：把"规则"和"真实 LLM"统一成同一套 plan / synthesize 接口。

设计原则（详见 README / docs/04）：
  - LLM 管"脑"：理解问法、决定调哪个工具、把知识与数据组织成自然语言答案。
  - 确定性代码管"手"：向量检索执行、真实数据计算（mcp_tools）、参数校验、输出结构。
  - 双后端兜底：没填 key 时 RuleBasedBackend 照常跑（零外部依赖，复用关键词版），
    填了 key 立即变真 LLM（OpenAIBackend）。任一 LLM 环节失败都优雅回退到规则。

后端选择（环境变量 LLM_BACKEND）：
  - rulebased / rules / keyword  -> RuleBasedBackend（默认）
  - openai / llm / gpt / deepseek / qwen / zhipu -> OpenAIBackend（OpenAI 兼容接口）
"""
from __future__ import annotations

import json
import os
import re
from typing import Any


# ---------------------------------------------------------------------------
# 统一接口契约
#   plan(query, tools) -> {
#       "intent": "rag" | "tool" | "hybrid",
#       "needs_knowledge": bool,            # 是否检索知识库
#       "tool_calls": [{"name", "arguments", "result"(规则后端预填, LLM 后端为 None)],
#   }
#   synthesize(query, contexts, tool_results, plan) -> {
#       "scenario": str, "findings": [str], "strategy": [str], "risks": [str]
#   }
# ---------------------------------------------------------------------------

class RuleBasedBackend:
    """确定性兜底后端：复用关键词版链路（route + dispatch_tool + 模板答案）。

    行为与原 supply-chain 关键词版完全一致（同一套函数、同一份评测口径），
    因此 21 条评测与冒烟测试在 LLM_BACKEND=rulebased 下 100% 复现。
    """

    name = "rulebased"

    def plan(self, query: str, tools: list[dict]) -> dict[str, Any]:
        from keyword_router import keyword_plan
        return keyword_plan(query)

    def synthesize(self, query: str, contexts: list[dict], tool_results: list[dict],
                   plan: dict) -> dict[str, Any]:
        from keyword_router import build_hybrid_answer, build_rag_answer, build_tool_answer

        intent = plan["intent"]
        tool_out = tool_results[0] if tool_results else {"tool": None, "result": {}}
        if intent == "rag":
            return build_rag_answer(query, contexts)
        if intent == "tool":
            return build_tool_answer(query, tool_out)
        return build_hybrid_answer(query, contexts, tool_out)

    def usage(self) -> dict[str, int] | None:
        return None


class OpenAIBackend:
    """真实 LLM 后端：OpenAI 兼容接口（OpenAI / DeepSeek / 通义 / 智谱 / 本地 Ollama）。

    两个插入点（取代关键词版的关键词匹配 + 模板填空）：
      ① plan  —— 用 function calling 让 LLM 自己选工具 + 抽参数；
      ② synthesize —— 用 LLM 把召回知识与工具结果综合成结构化 JSON 答案（防幻觉）。
    任一环节调用失败都会优雅回退到 RuleBasedBackend，保证 demo 永不崩。
    """

    name = "openai"

    def __init__(self) -> None:
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY 未设置，无法使用 OpenAI 后端。请设置环境变量或改用 "
                "LLM_BACKEND=rulebased。"
            )
        from openai import OpenAI
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
        )
        self.model = os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        self._usage: dict[str, int] | None = None

    # ----- plan：LLM 函数调用路由 + 参数抽取 -----
    def plan(self, query: str, tools: list[dict]) -> dict[str, Any]:
        system = (
            "你是供应链协同助手。判断用户问题需要调用哪些数据工具（如需要），"
            "以及是否需要检索知识库（SOP/公式）。只能从给定工具中选择，参数要准确。"
            "如果问题只涉及方法论/公式定义而不需要具体数据，就不要调用工具。"
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": query},
                ],
                tools=manifest_to_openai_tools(tools),
                tool_choice="auto",
                temperature=0,
            )
        except Exception:
            return RuleBasedBackend().plan(query, tools)

        self._usage = _usage_dict(resp)
        msg = resp.choices[0].message
        tool_calls: list[dict[str, Any]] = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            tool_calls.append({"name": tc.function.name, "arguments": args, "result": None})

        # 调工具时同时检索知识做混合（提升答案可靠性）；不调工具则纯知识问答
        needs_knowledge = True
        intent = "rag" if not tool_calls else "hybrid"
        return {"intent": intent, "needs_knowledge": needs_knowledge, "tool_calls": tool_calls}

    # ----- synthesize：LLM 生成结构化答案 -----
    def synthesize(self, query: str, contexts: list[dict], tool_results: list[dict],
                   plan: dict) -> dict[str, Any]:
        knowledge = "\n\n".join(
            f"【知识 {i + 1}】{c['content']}" for i, c in enumerate(contexts)
        ) or "（无相关知识点）"
        data = "\n".join(
            f"- {t['tool']}: {json.dumps(t['result'], ensure_ascii=False)[:600]}"
            for t in tool_results
        ) or "（无工具数据）"
        system = (
            "你是供应链策略分析师。基于下方【知识】和【数据】回答用户问题，不得编造未给出的信息。"
            "输出严格 JSON：{\"scenario\": str, \"findings\": [str], \"strategy\": [str], \"risks\": [str]}。"
            "findings 给出关键发现；strategy 给出可执行建议；risks 给出前提与口径风险。"
        )
        user = f"用户问题：{query}\n\n【知识】\n{knowledge}\n\n【数据】\n{data}"
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            self._usage = _merge_usage(self._usage, resp)
            obj = _extract_json(resp.choices[0].message.content)
            obj.setdefault("scenario", "供应链问答（LLM 生成）")
            obj.setdefault("findings", [str(query)])
            obj.setdefault("strategy", [])
            obj.setdefault("risks", [])
            return obj
        except Exception:
            return RuleBasedBackend().synthesize(query, contexts, tool_results, plan)

    def usage(self) -> dict[str, int] | None:
        return self._usage


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def manifest_to_openai_tools(tools: list[dict]) -> list[dict]:
    """把 mcp_tools.tool_manifest() 的宽松 schema 转成 OpenAI function-calling schema。"""
    out: list[dict] = []
    for t in tools:
        props: dict[str, dict] = {}
        required: list[str] = []
        for pname, spec in (t.get("input_schema") or {}).items():
            optional = str(spec).endswith("?")
            base = str(spec).rstrip("?")
            if base.startswith("int"):
                ptype = "integer"
            elif base.startswith("float") or base.startswith("number"):
                ptype = "number"
            elif base.startswith("bool"):
                ptype = "boolean"
            else:
                ptype = "string"
            props[pname] = {"type": ptype, "description": pname}
            if not optional:
                required.append(pname)
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        })
    return out


def coerce_args(args: dict[str, Any]) -> dict[str, Any]:
    """把 LLM 返回的参数做类型规整（字符串数字 -> int 等），避免调用工具时类型错。"""
    out: dict[str, Any] = {}
    for k, v in args.items():
        if isinstance(v, bool) or isinstance(v, int) or isinstance(v, float):
            out[k] = v
        elif isinstance(v, str):
            s = v.strip()
            if re.fullmatch(r"-?\d+", s):
                out[k] = int(s)
            else:
                out[k] = s
        else:
            out[k] = v
    return out


def _extract_json(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
    return {}


def _usage_dict(resp: Any) -> dict[str, int] | None:
    u = getattr(resp, "usage", None)
    if not u:
        return None
    return {
        "prompt_tokens": getattr(u, "prompt_tokens", 0),
        "completion_tokens": getattr(u, "completion_tokens", 0),
        "total_tokens": getattr(u, "total_tokens", 0),
    }


def _merge_usage(prev: dict[str, int] | None, resp: Any) -> dict[str, int] | None:
    cur = _usage_dict(resp)
    if not cur:
        return prev
    if not prev:
        return cur
    return {k: prev.get(k, 0) + cur.get(k, 0) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}


def get_backend(name: str | None = None):
    """按环境变量 LLM_BACKEND 选择后端（默认 rulebased）。"""
    name = (name or os.environ.get("LLM_BACKEND", "rulebased")).lower()
    if name in ("openai", "llm", "gpt", "gpt4", "deepseek", "qwen", "zhipu"):
        return OpenAIBackend()
    return RuleBasedBackend()
