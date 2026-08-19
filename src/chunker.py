# -*- coding: utf-8 -*-
"""
Markdown 切片器（chunker）· 第一性原理版
设计目标：检索的基本单位 = "一个自洽的作答信息单元"。
- 人类作者用标题已经把"一个想法"边界标好了，直接复用（按标题切），不为省事强行固定窗口。
- 每个 chunk 携带"来源路径"元数据，解决跨块引用断裂（模型看到「应用说明」也知道它属于哪个 SOP 的哪条公式）。
- 超长节回退滑动窗口（护栏），避免单块过大撑爆上下文。
"""
from __future__ import annotations

import re
from typing import Any

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def _est_chars(text: str) -> int:
    """中文为主，用字符数近似 token 预算（1 中文≈1~2 token，字符数足够做切片阈值）。"""
    return len(text)


def chunk_markdown(
    text: str,
    source: str,
    doc_title: str | None = None,
    heading_level: int = 2,
    max_chars: int = 800,
    overlap_chars: int = 80,
) -> list[dict[str, Any]]:
    """把一篇 md 切成 chunk 列表。

    - heading_level: 在此级（含）及以上标题处切分（默认 2 级 `##`）。
    - max_chars: 单块字符上限，超出则在该节内滑动窗口回退。
    - overlap_chars: 滑动窗口重叠，缓解句子被切断。
    返回字段：id, source, doc_title, section_path, title, content
    """
    lines = text.splitlines()
    # 取 H1 作为文档标题（若未显式传入）
    if doc_title is None:
        for ln in lines:
            m = HEADING_RE.match(ln)
            if m and len(m.group(1)) == 1:
                doc_title = m.group(2).strip()
                break
    doc_title = doc_title or source

    chunks: list[dict[str, Any]] = []
    buf: list[str] = []
    cur_title = doc_title

    def flush():
        if not buf:
            return
        body = "\n".join(buf).strip()
        if not body:
            return
        # 超长节 -> 滑动窗口回退
        if _est_chars(body) > max_chars:
            for i, seg in enumerate(_slide(body, max_chars, overlap_chars)):
                _emit(seg, f"{cur_title}（续{i+1}）", suffix=f"-p{i+1}")
        else:
            _emit(body, cur_title, suffix="")

    def _emit(body: str, title: str, suffix: str):
        section_path = f"{doc_title} > {title}"
        # 来源路径前缀：即便块被单独召回，模型也知道它属于哪个 SOP 的哪条
        wrapped = f"[来源: {section_path}]\n{body}"
        chunks.append(
            {
                "id": f"{source}-{len(chunks)+1}{suffix}",
                "source": source,
                "doc_title": doc_title,
                "section_path": section_path,
                "title": title,
                "content": wrapped,
            }
        )

    for ln in lines:
        m = HEADING_RE.match(ln)
        if m and len(m.group(1)) <= heading_level:
            flush()
            cur_title = m.group(2).strip()
            buf = [ln]
        else:
            buf.append(ln)
    flush()
    return chunks


def _slide(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    step = max(max_chars - overlap_chars, 1)
    segs = []
    for start in range(0, len(text), step):
        segs.append(text[start : start + max_chars])
        if start + max_chars >= len(text):
            break
    return segs
