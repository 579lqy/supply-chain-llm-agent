# -*- coding: utf-8 -*-
"""演示入口：跑通一条真实（非编造）的混合问法。

运行方式（零依赖，默认走关键词兜底后端）：
    python src/demo.py

接入真 LLM（OpenAI 兼容，例如 DeepSeek / 通义 / 智谱 / 本地 Ollama）：
    set LLM_BACKEND=openai
    set OPENAI_API_KEY=sk-xxx
    set OPENAI_BASE_URL=https://api.openai.com/v1   # 或厂商地址
    set OPENAI_MODEL=gpt-4o-mini
    python src/demo.py

把语义检索升级为向量（无需 key）：
    pip install sentence-transformers   # 首次会自动下载 bge-small-zh 模型
    python src/demo.py                  # 自动从 TF-IDF 回退切换为向量检索
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_agent import LlmRagAgent, format_answer  # noqa: E402

DEMO_QUERY = "Mumbai 仓哪个 SKU 最该优先补货（结合库存覆盖与缺陷风险）？"


def main() -> None:
    backend_name = os.environ.get("LLM_BACKEND", "rulebased")
    print(f"[后端] {backend_name}  ｜ 演示问法：{DEMO_QUERY}\n")
    agent = LlmRagAgent()
    answer = agent.run(DEMO_QUERY)
    print(format_answer(answer))


if __name__ == "__main__":
    main()
