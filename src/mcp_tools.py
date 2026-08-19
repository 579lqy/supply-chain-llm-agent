from __future__ import annotations

import csv
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chunker import chunk_markdown


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
KB_DIR = ROOT / "knowledge_base"
CSV_NAME = "clean_supply_chain.csv"

# 真实公开数据列（clean_supply_chain.csv, 100 行 × 24 列）
REAL_COLUMNS = [
    "product_type", "sku", "price", "availability", "units_sold", "revenue",
    "customer_demographics", "stock_level", "lead_times_days", "order_qty",
    "shipping_time_days", "shipping_carrier", "shipping_cost", "supplier_name",
    "location", "lead_time_days", "production_volume", "mfg_lead_time_days",
    "mfg_cost", "inspection_result", "defect_rate", "transport_mode",
    "route", "total_cost",
]


def read_csv(name: str = CSV_NAME) -> list[dict[str, str]]:
    path = DATA_DIR / name
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def to_int(value: str) -> int:
    return int(float(value or 0))


def to_float(value: str) -> float:
    return float(value or 0)


@dataclass
class SkuFacts:
    sku: str
    product_type: str
    price: float
    units_sold: int
    revenue: float
    stock_level: int
    lead_times_days: float       # 报价提前期（与 lead_time_days 不同，已核对两列几乎零相关）
    lead_time_days: float        # 实际提前期
    shipping_cost: float
    mfg_cost: float
    total_cost: float
    supplier_name: str
    location: str
    defect_rate: float
    inspection_result: str


def load_knowledge_documents() -> list[dict[str, str]]:
    """加载知识库 md 并切成 chunk（含来源路径元数据，解决跨块引用断裂）。"""
    docs: list[dict[str, str]] = []
    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.extend(chunk_markdown(text, source=path.name))
    return docs


# ----------------------------------------------------------------------------
# 真实数据工具：所有数值/统计问答直接查真实行，不经过文本检索（架构分工）
# ----------------------------------------------------------------------------

def find_sku(sku: str) -> SkuFacts | None:
    for row in read_csv():
        if row["sku"].lower() == sku.lower():
            return SkuFacts(
                sku=row["sku"],
                product_type=row["product_type"],
                price=to_float(row["price"]),
                units_sold=to_int(row["units_sold"]),
                revenue=to_float(row["revenue"]),
                stock_level=to_int(row["stock_level"]),
                lead_times_days=to_float(row["lead_times_days"]),
                lead_time_days=to_float(row["lead_time_days"]),
                shipping_cost=to_float(row["shipping_cost"]),
                mfg_cost=to_float(row["mfg_cost"]),
                total_cost=to_float(row["total_cost"]),
                supplier_name=row["supplier_name"],
                location=row["location"],
                defect_rate=to_float(row["defect_rate"]),
                inspection_result=row["inspection_result"],
            )
    return None


def low_stock_skus(location: str | None = None, threshold: int | None = None,
                   top_n: int | None = None) -> list[dict[str, Any]]:
    rows = read_csv()
    if location:
        rows = [r for r in rows if r["location"].lower() == location.lower()]
    if threshold is not None:
        rows = [r for r in rows if to_int(r["stock_level"]) < threshold]
    rows.sort(key=lambda r: to_int(r["stock_level"]))
    selected = rows[:top_n] if top_n else rows
    return [
        {"sku": r["sku"], "location": r["location"], "stock_level": to_int(r["stock_level"]),
         "defect_rate": to_float(r["defect_rate"])}
        for r in selected
    ]


def supplier_defect_ranking() -> list[dict[str, Any]]:
    rows = read_csv()
    by_supplier: dict[str, list[float]] = {}
    for r in rows:
        by_supplier.setdefault(r["supplier_name"], []).append(to_float(r["defect_rate"]))
    ranked = [
        {"supplier": s, "avg_defect_rate": round(statistics.mean(v), 4), "n": len(v)}
        for s, v in by_supplier.items()
    ]
    ranked.sort(key=lambda x: x["avg_defect_rate"], reverse=True)
    return ranked


def avg_lead_time_by_location() -> list[dict[str, Any]]:
    rows = read_csv()
    by_loc: dict[str, list[float]] = {}
    for r in rows:
        by_loc.setdefault(r["location"], []).append(to_float(r["lead_time_days"]))
    return [
        {"location": loc, "avg_lead_time_days": round(statistics.mean(v), 2), "n": len(v)}
        for loc, v in by_loc.items()
    ]


def lead_time_stats() -> dict[str, float]:
    vals = [to_float(r["lead_time_days"]) for r in read_csv()]
    mean = statistics.mean(vals)
    std = statistics.pstdev(vals)
    return {"mean": round(mean, 2), "std": round(std, 2), "mean_plus_2std": round(mean + 2 * std, 2)}


def abc_classification() -> dict[str, Any]:
    """按 revenue 降序累计，A≤80% / B≤95% / C 其余（标准 ABC 法）。"""
    rows = sorted(read_csv(), key=lambda r: to_float(r["revenue"]), reverse=True)
    total = sum(to_float(r["revenue"]) for r in rows)
    cum = 0.0
    ordered = []
    for r in rows:
        rev = to_float(r["revenue"])
        cum += rev
        ordered.append({
            "sku": r["sku"], "revenue": round(rev, 2),
            "cum_pct": round(cum / total * 100, 2),
        })
    # 标注类别
    for i, item in enumerate(ordered):
        item["class"] = "A" if item["cum_pct"] <= 80 else ("B" if item["cum_pct"] <= 95 else "C")
    # top 20% SKU 贡献的营收占比
    top20_n = max(1, int(len(ordered) * 0.2))
    top20_rev = sum(to_float(r["revenue"]) for r in rows[:top20_n])
    return {
        "ordered": ordered,
        "top20_sku_share": round(top20_n / len(ordered) * 100, 1),
        "top20_revenue_share": round(top20_rev / total * 100, 1),
    }


def supplier_scorecard(sku: str) -> dict[str, Any]:
    """质量=1-缺陷率（AQL 2.5% 判定）；交付=实际提前期是否低于 mean+2σ；成本=运费+制造成本。"""
    facts = find_sku(sku)
    if facts is None:
        raise ValueError(f"未找到 SKU: {sku}")
    stats = lead_time_stats()
    quality_pass = facts.defect_rate <= 0.025
    delivery_pass = facts.lead_time_days <= stats["mean_plus_2std"]
    quality_score = round((1 - facts.defect_rate) * 100, 1)
    return {
        "sku": sku,
        "supplier": facts.supplier_name,
        "defect_rate": facts.defect_rate,
        "quality_pass": quality_pass,
        "quality_score": quality_score,
        "lead_time_days": facts.lead_time_days,
        "delivery_baseline": stats["mean_plus_2std"],
        "delivery_pass": delivery_pass,
    }


def tool_manifest() -> dict[str, Any]:
    return {
        "tools": [
            {"name": "find_sku", "description": "查询某 SKU 的真实字段（库存/缺陷率/提前期/成本等）",
             "input_schema": {"sku": "string"}},
            {"name": "low_stock_skus", "description": "按城市/阈值筛选低库存 SKU，可限制返回条数",
             "input_schema": {"location": "string?", "threshold": "int?", "top_n": "int?"}},
            {"name": "supplier_defect_ranking", "description": "按供应商聚合平均缺陷率并排序",
             "input_schema": {}},
            {"name": "avg_lead_time_by_location", "description": "按城市聚合平均实际提前期",
             "input_schema": {}},
            {"name": "lead_time_stats", "description": "全数据集实际提前期的均值与均值+2σ",
             "input_schema": {}},
            {"name": "abc_classification", "description": "按营收做 ABC 分类并给出 top20% 营收占比",
             "input_schema": {}},
            {"name": "supplier_scorecard", "description": "对指定 SKU 的供应商做质量/交付评分卡",
             "input_schema": {"sku": "string"}},
        ]
    }
