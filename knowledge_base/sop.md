# 供应商质量与风险 SOP（基于质量管理标准）

> 本节为可引用的领域知识（标准方法论），用于 RAG 检索；具体数值由工具层（MCP）从真实数据集实时计算。

## SOP-1 缺陷率判定

- **缺陷率定义**：`缺陷率 = 不合格单位数 / 检验单位总数`（0–1，或折算为 PPM = 缺陷率 × 10⁶）。
- **可接受质量限 AQL**：按 ANSI/ASQ Z1.4（等同 ISO 2859-1）一般检验水平 II，**AQL = 2.5%** 为消费品通用接收上限；缺陷率 > AQL 判为"质量不达标"。
- 执行：取该供应商在真实数据集中的 `defect_rate`，与 AQL 2.5% 比较输出"达标 / 不达标"。

## SOP-2 供应商综合评分卡

维度与权重（行业通行实践）：

- 质量（缺陷率/PPM）：权重 ~40%
- 交付（准时率 OTIF = 按期交付量 / 总交付量）：权重 ~35%
- 成本（相对基准成本指数）：权重 ~25%

判定：质量不达标（>AQL）或 OTIF < 90% 任一触发，即列为**高风险供应商**，进入替代评估流程。

## SOP-3 高风险供应商处理

1. 确认高风险 SKU、缺口量与可恢复日期。
2. 查询替代供应商的准时率、缺陷率与 MOQ。
3. 评估切换供应商或拆分采购订单的可行性。
4. 同步计划与销售重评客户交付承诺，写入供应商绩效记录。

## 引用来源

- ANSI/ASQ Z1.4-2003 (R2018). *Sampling Procedures and Tables for Inspection by Attributes*. —— AQL 2.5% 一般检验水平 II 接收限。
- Heizer, J., Render, B., & Munson, C. (2020). *Operations Management* (13th ed.). Pearson. —— 质量管理与供应商评分（第 6、17 章）。
- APICS / ASCM. *APICS Dictionary* (16th ed.). —— OTIF、PPM、supplier scorecard 定义。
