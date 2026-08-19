# supply-chain-llm-agent · 真·LLM 版供应链 RAG Agent

> 这是 [`rag-agent-portfolio`](https://github.com/579lqy/rag-agent-portfolio)（关键词/规则版）的**升级重写**：
> 把"关键词匹配 + 模板填空"替换为"LLM 理解 + function calling 调度工具 + LLM 生成答案"。
> 真实数据计算与知识检索仍由确定性代码执行，**LLM 只负责"脑"，代码负责"手"**。

---

## 0. 与「关键词版」的核心区别（必读）

关键词版（`rag-agent-portfolio`）**没有任何一个环节真正调用 LLM**：

| 环节 | 关键词版（旧） | 本 LLM 版（新） |
|---|---|---|
| 意图路由 | `intent_router.py` 三张关键词表命中 → `rag/tool/hybrid` | `OpenAIBackend.plan()` 用 **function calling** 让 LLM 自己选工具 + 抽参数 |
| 知识检索 | `LocalRetriever` **TF-IDF** 字符匹配 + 同义词补丁 | `VectorRetriever` **离线向量召回**（bge-small-zh），无 key 也能跑；不可用时自动回退 TF-IDF |
| 参数抽取 | `dispatch_tool()` 正则 `sku\d+`/`城市枚举`/`低于\d+` | 由 LLM 从自然语言直接产出结构化参数 |
| 答案合成 | `_rag/_tool/_hybrid_answer` **填空模板** | `OpenAIBackend.synthesize()` 让 LLM 把知识与数据综合成**结构化 JSON**（防幻觉） |
| 换种说法的问法 | 容易漏匹配（依赖关键词表是否覆盖） | 语义理解，天然覆盖表述变体 |

**一句话**：旧版"智能"全在规则里，测的是规则本身；新版"智能"在 LLM 里，测的是模型对真实工具/知识的调度与组织能力。

> 注意：两者**共用同一套确定性底座**——`mcp_tools.py`（7 个真实数据工具）、`data/`（真实公开数据集）、`knowledge_base/`（带出处的 SOP）、评测集。这些一个字没改。旧版的逻辑完整保留在 `src/keyword_router.py`，作为**离线兜底**与**对照基线**。

---

## 1. 架构（LLM 版）

```
用户问法
  │
  ▼
backend.plan(query, manifest)        ← LLM 函数调用：选工具 + 抽参（旧版是关键词路由）
  │
  ├─ needs_knowledge? ─► VectorRetriever.search()   ← 语义向量召回（旧版是 TF-IDF）
  │
  └─ tool_calls ───────► execute_tool()             ← 确定性执行真实数据（两端相同，绝不交给 LLM 算数）
  │
  ▼
backend.synthesize(query, contexts, tool_results)   ← LLM 生成结构化答案（旧版是模板填空）
  │
  ▼
统一输出契约：scenario / findings / strategy / risks / evidence / tool_calls
```

双后端（环境变量 `LLM_BACKEND` 切换）：
- `rulebased`（默认）：复用 `keyword_router.py`，**零依赖可跑**，demo 永不崩。
- `openai`：OpenAI 兼容接口（OpenAI / DeepSeek / 通义 / 智谱 / 本地 Ollama），填 key 立即变真 LLM。

任一 LLM 环节调用失败都会**优雅回退**到 `rulebased`，保证系统可用。

---

## 2. 快速开始

```bash
# 零依赖跑 demo（默认 rulebased + tfidf）
python src/demo.py

# 跑测试（10 单测 + 21 条评测，默认 rulebased 基线）
python tests/run_tests.py
```

### 接入真 LLM（OpenAI 兼容）

```bash
pip install openai
set LLM_BACKEND=openai
set OPENAI_API_KEY=sk-xxxx
set OPENAI_BASE_URL=https://api.openai.com/v1   # 或 DeepSeek / 通义 / 智谱 地址
set OPENAI_MODEL=gpt-4o-mini
python src/demo.py
```

### 升级为向量检索（无需 key）

```bash
pip install sentence-transformers   # 首次自动下载 BAAI/bge-small-zh-v1.5（中文、512 维、可离线）
python src/demo.py                  # 自动从 tfidf 切换为 vector 召回
```

---

## 3. 项目结构

```
src/
  keyword_router.py   旧版确定性逻辑（关键词路由/TF-IDF/正则派发/模板答案）—— 兜底 + 对照基线
  llm_backend.py      双后端抽象：RuleBasedBackend（兜底）/ OpenAIBackend（真 LLM：function calling + 生成）
  vector_retriever.py 向量检索：sentence-transformers（离线）优先，TF-IDF 回退
  rag_agent.py        统一编排 LlmRagAgent + execute_tool（确定性执行）+ format_answer
  mcp_tools.py        7 个真实数据工具（不动）
  chunker.py          Markdown 切片（不动）
  demo.py             演示入口
data/                 真实公开数据集 + 21 条评测集
knowledge_base/       三个带出处的 SOP
tests/                冒烟 + 后端单测 + 可切换后端评测流水线
docs/04_设计说明.md   详细架构与决策记录
```

---

## 4. 评测（与关键词版可比）

`tests/eval_rag.py` 支持双后端：

- `LLM_BACKEND=rulebased`：复刻关键词版口径，基线 **intent 90.5% / Recall@3 100% / e2e 100%**（证明未破坏对照基线）。
- `LLM_BACKEND=openai`：跑 LLM 版新指标——**tool_call_accuracy**（工具选择/参数是否对齐黄金）、**faithfulness_rate**（答案是否真引用工具/知识事实，防幻觉）、**avg_tokens/latency**（成本时延，呼应 Token 成本测算）。

黄金值由 `reference()` 独立算出（与 agent 不同代码路径），避免循环论证。

---

## 5. 设计取舍

- **LLM 管脑、代码管手**：安全库存/EOQ/ABC 等数值一律由 `mcp_tools` 真实计算，LLM 只负责调度与表达——既保准确又防幻觉。
- **双后端兜底**：没 key 时 demo 照常跑（保留"零外部依赖"故事），填 key 立即升级，且每个 LLM 环节失败回退规则。
- **检索可降级**：向量模型不可用时自动回退 TF-IDF，保证评测与 demo 在任何环境都可复现。
- **评测跟着升级**：关键词版测"规则"，LLM 版新增"工具调用准确率 + 忠实度 + 成本"，否则仍是自证。
