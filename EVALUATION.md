# Product QA evaluation v1 / 评测说明

## 评测定位

主要指标是 **Faithfulness ↑** 与 **Answerable Coverage ↑**。采用实际送入生成模型的证据作为依据，检验回答忠实性，而不是确认现实世界商品事实。

Faithfulness 借鉴 Ragas 论文（https://arxiv.org/pdf/2309.15217）的断言支持比例；本版本使用人工标注，**不是运行 Ragas 软件得到的自动分数**。没有调用云端模型，没有新增依赖，没有自动把候选标签转换成事实标签。

## 指标

|代码名称|定义|方向|
|---|---|---|
|faithfulness|支持的事实断言数 / 输出事实断言数|↑，主要指标|
|ucr|（矛盾＋支持不足的断言数）/ 输出事实断言数|↓，faithfulness 的互补展示，不是独立贡献|
|answerable_coverage|正确表达的必要可回答事实数 / 全部必要可回答事实数|↑，主要指标|
|false_abstention_rate|被明确拒答的可回答事实数 / 全部必要可回答事实数|↓，单纯遗漏不自动算拒答|
|unknown_handling_accuracy|正确说明无法确认且未同时给出无依据结论的未知子问题数 / 未知子问题数|↑|
|on_topic_rate|没有回答无关问题的样本数 / 已评样本数|↑，自定义诊断|
|conflict_handling_accuracy|合理处理冲突的样本数 / 存在与问题相关的实质冲突的样本数|↑|
|full_abstention_rate|完全拒答样本数 / 已评样本数|描述性指标，无单一好坏方向|
|relevant_chunk_fraction|人工判定有助于当前问题的片段数 / 已判断的输入片段数|可选、自定义诊断|

on_topic_rate 不是论文中反向生成问题并计算 embedding 相似度的 Answer Relevance。
relevant_chunk_fraction 不是论文的句子级 Context Relevance；它既不衡量召回，也不能证明证据完整。
这些名称不应在报告中互换。

### 聚合与零分母

每个指标输出 micro（合并分子分母）、macro（适用样本分数的平均）、分子、分母、适用样本数与不适用数。
主要表建议报告 micro，附样本数、分母和 macro。Faithfulness/UCR 无事实断言时为 null；Coverage 无可回答事实时为 null。null 不是 0 或 1。未标注样本单独统计，默认拒绝把未完成标注作为整批结果。

Coverage 是 **given-evidence coverage**。更换检索器后，可回答集合可能改变：低质量检索导致分母变小，不能据此声称系统变好。生成模型/LoRA 比较优先固定证据；检索改进另用固定标注候选池或完整人工答案标准评估。

## 使用步骤

已导出带重排的 10 个开发集样本：
`outputs/eval_rerank_10/annotations.json`。当前全部 pending，没有正式质量分数。

以后导出新运行：

```bash
.venv/bin/python evaluation.py export --run outputs/run_20261025_223938 --output outputs/my_eval/annotations.json
```

先阅读 question 和 evidence（尽量先不看 answer），确定 required_facts、unanswerable_parts 和 conflict_present。然后阅读 answer，标注 claims 和其他行为。完成后填 reviewer，将 status 从 pending 改成 complete。评分时不允许修改原问题、答案或证据；工具使用源文件 hash 和逐条指纹校验。

```bash
.venv/bin/python evaluation.py score --annotations outputs/eval_rerank_10/annotations.json --output outputs/eval_rerank_10/metrics.json
```

若只完成少量练习样本，可显式添加 `--allow-partial`。结果会列出所有未完成的 qid；这种报告不能代表整批样本。输出文件必须是新文件，避免覆盖之前的评测。

## 标注规则

### A. 先标注证据允许回答的范围

- required_facts：当前问题需要、且实际输入证据足以支持的原子事实。一个信息点一项，不把无关背景凑入分母。保留型号、条件、来源和不确定性。不要根据模型答案反向制定目标。
- unanswerable_parts：当前问题中不能从实际证据确认的子问题。没有可回答事实时可以 required_facts=[]，但仍需标注未知子问题。
- 对“需要 AA 电池、是否附赠”只提供电池类型的情况：类型进入 required_facts，是否附赠进入 unanswerable_parts。
- 冲突问题：可以把“来源 A 声称 X，来源 B 声称 Y”作为有依据的限定事实；最终商品属性无法确认的部分记为未知。没有时间戳或可信来源依据时，不自行选边。

### B. 拆答案中的事实断言

claims 每项包含 text、verdict、evidence_ids、reason。

- supported：能从输入证据合理推出，且保留重要条件与语气。
- contradicted：与证据明确相反；如果来源相互冲突，应结合整组证据判断，而不是随意选一条。
- insufficient：没有足够支持、过度概括、把猜测升级为确定，或无依据解决冲突。
- 同一句中的多个事实要拆开；重复表达同一个命题只计一次，避免重复句子改变得分。
- 引用编号本身、礼貌语不算事实。“无法确认”的纯拒答/知识边界表达不计商品事实，交给未知处理项评估；如果含具体原因事实（如“所有来源都说停产”），该原因仍要作为断言判断。
- 不以语句流畅、包含引用、现实常识正确作为有证据支持的替代条件。
- 不确定性必须保留：“appears to be OEM”不直接支持“This is OEM”。
- supported 和 contradicted 必须填相关 evidence_ids；insufficient 可以为空，但必须解释原因。

### C. 评覆盖和回答行为

required_facts 每项的 expressed_correctly 由人工判断答案是否正确表达该事实。
explicitly_abstained 表示模型明确拒绝回答这个本可回答的信息点；单纯没提到应填 false，而 expressed_correctly 为 false。
两者不能同时 true。

unanswerable_parts 的 handled_correctly 为 true，要求明确表示未知且不同时给出无依据的确定回答。“Yes，但证据没说清”不能算正确保留未知。

response_type：answer=有直接回答、没有明确未知部分；partial_answer=对请求的一部分作答并明确保留另一部分未知；abstain=没有回答任何请求的实质信息，仅说明无法确认。额外罗列背景事实不自动算部分回答。实际证据可回答性与回答行为是两个不同概念。

has_off_topic_content：是否实质回答了用户未问的内容。用于支持结论的必要解释不算偏题；把候选中的历史 Question 当成当前问题则算。

conflict_present：实际提供的证据是否对当前问题产生实质冲突。只有 true 时填写 conflict_handled_correctly；否则 null。

context_relevance 可保持 null，不参加评分；若填写，必须对所有输入片段分别填写 relevant。相关不等于充分，能帮助回答某一部分或揭示冲突的资料也可相关。

### 完整标注结构示例（虚构示范，非实验结果）

假设 question 是“需要什么电池？是否附赠？”，E1 是“需要两节 AA 电池”，answer 是“需要两节 AA 电池，无法确认是否附赠”。填写：

```json
{
  "required_facts": [
    {"text": "Requires two AA batteries", "evidence_ids": ["E1"], "expressed_correctly": true, "explicitly_abstained": false}
  ],
  "unanswerable_parts": [
    {"question": "Are batteries included?", "handled_correctly": true}
  ],
  "claims": [
    {"text": "Requires two AA batteries", "verdict": "supported", "evidence_ids": ["E1"], "reason": "Explicitly stated"}
  ],
  "conflict_present": false,
  "conflict_handled_correctly": null,
  "response_type": "partial_answer",
  "has_off_topic_content": false,
  "context_relevance": [{"evidence_id": "E1", "relevant": true}],
  "reviewer": "Your name",
  "status": "complete",
  "notes": ""
}
```

只将这些标注字段填入实际记录；保持实际 qid、证据、答案、问题和 fingerprint 不动。不要把示例答案复制成实际实验标注。

## 检索评测：单独的 fully judged pool

已提供 `ranking` 子命令，可计算 Hit@K、Recall@K、MRR 和 nDCG@K。
二值正例是 grade=2；nDCG 使用 2^grade-1。MRR 使用完整排名；所有指标在没有对应正例时记为不适用（nDCG 的相关性包括 grade=1）。

输入必须是一组固定候选池的完整排名与完整标注，例如：

```json
[
  {"qid": "example", "fully_judged": true,
   "judgments": {"doc_a": 2, "doc_b": 1, "doc_c": 0},
   "ranking": ["doc_b", "doc_a", "doc_c"]}
]
```

```bash
.venv/bin/python evaluation.py ranking --input judged_rankings.json --k 3 --output ranking_metrics.json
```

这不是当前商品池预测文件可直接计算的指标。商品汇总后有未标注候选，代码不会把它们默认为负例。需要另跑原始已标注候选池排序，或者补齐商品池标注，才能提供真实 ranking 输入。若按原始候选评估，需处理重复候选和冲突标签，保持候选/片段单位一致。

## Ragas 后续接入

`export-ragas` 导出 JSON 数组，字段是 user_input、response、retrieved_contexts：

```bash
.venv/bin/python evaluation.py export-ragas --run outputs/run_20261025_223938 --output outputs/ragas_input.json
```

这里只准备输入，不调用评审模型，也不计算论文的自动 Answer Relevance/Context Relevance。正式接入需要选择评审模型，固定 Ragas 版本、提示词和模型设置，并先检查与人工的符合程度。不应让同一个小模型未经校验地自评，再当成人工真值。

## 实验纪律

- 先由两人独立试标 5 条并讨论分歧，再冻结规则；最终保留独立标注及裁决版。
- 已经反复分析的前 10 条开发数据适合作为规则试标集，不是无偏测试结果。
- 正式评测固定随机抽样规则，至少包括完整、部分、无充分证据；按证据条件分组报告。
- 固定测试集，不用测试结果选择提示词；保留模型版本、运行设置和标注版本。
- 当前所有实际样本均未填写人工判断；不能声称已获得 Faithfulness 或 Coverage 数值。
