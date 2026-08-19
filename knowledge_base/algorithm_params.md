# 订货批量与分类模型（基于经典运营管理）

> 本节为可引用的领域知识（标准方法论），用于 RAG 检索；具体数值由工具层（MCP）从真实数据集实时计算。

## 1. 经济订货批量 EOQ（Economic Order Quantity）

使"订货成本 + 持有成本"之和最小的每次订货量：

`EOQ = √(2 × D × S / H)`

- `D`：年需求量（单位/年）。
- `S`：每次订货（或换线/setup）成本。
- `H`：单位商品年持有成本（= 单位成本 × 持有成本率）。

原理：订货频次↑ → S 成本↑但持有成本↓；EOQ 取二者拐点。

## 2. ABC 分类（帕累托库存分类）

按**年消耗金额**（`年需求量 × 单位成本`）降序排列，累计占比分层：

- **A 类**：约占年消耗金额 70%–80%，SKU 数约 10%–20% → 重点管控（高频复盘、严安全库存）。
- **B 类**：约占金额 15%，SKU 约 30% → 中等管控。
- **C 类**：约占金额 5%–10%，SKU 约 50%–60% → 宽松管控（高 ROP、低复盘频次）。

## 3. 应用说明

- EOQ 假设需求稳定、瞬时补货；真实快消品需求波动大，常与安全库存模型联用。
- ABC 决定"管控力度"，应作为补货优先级与盘点频率的依据，而非绝对阈值。

## 引用来源

- Harris, F. W. (1913). "How Many Parts to Make at Once." *Factory: The Magazine of Management*. —— EOQ 原始模型。
- Heizer, J., Render, B., & Munson, C. (2020). *Operations Management* (13th ed.). Pearson. —— 第 12 章（EOQ、ABC）。
- APICS / ASCM. *APICS Dictionary* (16th ed.). —— ABC analysis 定义与分层口径。
