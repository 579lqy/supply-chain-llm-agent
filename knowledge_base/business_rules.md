# 库存策略与补货规则（基于经典供应链管理）

> 本节为可引用的领域知识（标准方法论），用于 RAG 检索；具体数值由工具层（MCP）从真实数据集实时计算。

## 1. 安全库存 Safety Stock（SS）

应对需求与提前期不确定性的缓冲库存。

- **提前期稳定、需求波动**（最常用）：

  `SS = Z × σ_d × √L`

  - `Z`：服务水平系数（周期服务水平 CSL）。Z=1.65 → 95%，Z=2.33 → 99%（标准正态分位数）。
  - `σ_d`：单位周期需求标准差。
  - `L`：提前期（周期数）。

- **需求与提前期同时波动**（更完备）：

  `SS = Z × √(L × σ_d² + d̄² × σ_L²)`

  - `d̄`：平均周期需求；`σ_L`：提前期标准差。

## 2. 再订货点 Reorder Point（ROP）

`ROP = d̄ × L + SS`

当库存降至 ROP 时触发补货，使补货到货时库存恰好覆盖提前期内的需求 + 安全缓冲。

## 3. 应用说明

- 服务水平（Z）越高，SS 越大、持有成本越高——需在**缺货成本 vs 持有成本**间权衡。
- 真实数据中 `lead_time_days`、`stock_level`、`defect_rate` 等由工具层提供；本节只定义"应如何计算与判定"。

## 引用来源

- Chopra, S., & Meindl, P. (2021). *Supply Chain Management: Strategy, Planning, and Operation* (8th ed.). Pearson. —— 第 3 章 Managing Uncertainties in a Supply Chain（安全库存与 ROP 公式）。
- Heizer, J., Render, B., & Munson, C. (2020). *Operations Management: Sustainability and Supply Chain Management* (13th ed.). Pearson. —— 第 11 章（安全库存/ROP）。
- APICS / ASCM. *APICS Dictionary* (16th ed.). —— safety stock、reorder point 定义。
