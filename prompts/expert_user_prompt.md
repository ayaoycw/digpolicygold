# Enterprise Policy Matching — User Prompt Template

我需要对以下企业进行政策匹配分析。请根据你的【专家认知框架】，对该企业的工商数据进行深度特征工程，并生成 Deep Research 的搜索策略。

## 输入数据说明

以下字段会根据可用性动态填充。缺失的字段请跳过对应维度的分析。

### 必填字段
- **企业名称**：`{name}`
- **行业**：`{industry}`
- **地区**：`{region}`

### 空间载体分析字段
- **注册地址（全文）**：`{address}`

### 产业链分析字段
- **经营范围**：`{business_scope}`
- **注册资本**：`{registered_capital}`
- **知识产权**：发明专利 `{ip_invention}` 件，实用新型 `{ip_utility}` 件，软件著作权 `{ip_software}` 件

### 身份属性分析字段
- **股东信息**：
{shareholders_text}

### 人力资源动态字段
- **参保人数历史**：
{headcount_text}
- **员工规模**：`{employees}`
- **企业标签**：`{tags}`
- **成立时间**：`{founded}`

### 合规熔断字段
- **风险信息**：`{risk_info}`

## 输出要求

请按以下步骤输出：

### 1. 【特征逆向工程】
提取显性字段，并推导隐性维度（例如：从地址推导出它属于哪个园区/功能区）。

### 2. 【维度差距分析】
分析该企业在"钱（补贴）、权（资质）、人（人才）、规（合规）"四个维度的潜质。

### 3. 【Deep Research 搜索策略】
生成具体的搜索关键词组合（Search Queries）。
- 格式要求：`[维度]：[搜索意图] -> [具体关键词组合]`
- 每个维度至少 1-3 个搜索词
- 搜索词应适合在 Bing/百度 上直接搜索

### 4. 【构建 Long Context 指南】
告诉下一步的搜索模型，应该重点关注哪些细则（如："重点查找该园区的房租补贴门槛"）。

## 最终输出格式（严格 JSON）

```json
{{
  "feature_engineering": {{
    "spatial": "空间载体分析结果",
    "industry_chain": "产业链地位分析",
    "identity": "身份属性分析",
    "hr_dynamics": "人力资源动态分析",
    "compliance": "合规状态分析"
  }},
  "gap_analysis": {{
    "money": "补贴潜力评估",
    "qualification": "资质潜力评估",
    "talent": "人才政策潜力",
    "compliance": "合规风险评估"
  }},
  "tasks": [
    {{
      "dimension": "空间载体|产业链|身份属性|人力资源|合规|税收与财务|人才激励",
      "layer": "产业专项|税收优惠|资质认定|人才激励|用工补贴",
      "search_term": "具体搜索关键词",
      "priority": "high|medium|low",
      "reason": "搜索意图说明",
      "focus_hints": "给搜索模型的重点关注指引"
    }}
  ],
  "compliance_veto": {{
    "passed": true,
    "risk_level": "none|low|medium|high|blocked",
    "detail": "合规判断说明"
  }}
}}
```
