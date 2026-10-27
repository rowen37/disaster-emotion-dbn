# 行为 span 标注口径

## 任务定义

模型只完成两个相互配套的输出：帖子是否包含目标行为的概率 `P(behavior=true)`，以及零个或多个
`behavior_spans`。`behavior_types` 仅用于标注审计和后续词频分组，不作为训练目标，也不表示行为的
危害程度。

目标行为是灾害发生前后，由个人、群体或机构实施并与避险、求助、救援、应急管理、灾害信息流动
或社会秩序直接相关的动作。单纯灾情状态、自然后果、情绪和缺少灾害关联的普通日常活动不标注。

## 功能类型

| 类型 | 纳入范围 | 理论对应 |
|---|---|---|
| 防护避险 | 备灾、撤离、转移、避难、灾害相关物资准备 | Stowe 等的 Action、Preparation、Movement |
| 求助报险 | 求救、报告险情、说明被困位置、提出明确需求 | CrisisBench 的 Requests or needs；NIST 的 Anxious |
| 救援互助 | 搜救、抢险、志愿服务、捐赠、照护和物资援助 | CrisisBench 的 Donation and volunteering、Response efforts；NIST 的 Helpers |
| 应急管理 | 调度、管制、停运停课、封路、机构组织转移 | CrisisBench 的 Response efforts |
| 信息行为 | 报告、发布、传播、核实、提醒和定位等信息动作 | Stowe 等的 Reporting、Information；NIST 的 Detectives |
| 失范扰序 | 造谣、诈骗、哄抬、盗抢、阻碍救援、违规闯入、破坏和污染 | NIST 的 Exploiters、Manipulators |

同一帖子可以包含多个 span；每个 span 对齐一个 type。类型按行为功能划分，不按“轻微/严重”划分。

## span 边界

1. 标注原文中最小但可辨认的连续动作片段，通常为 2–6 个汉字；完整帖子始终作为模型输入。
2. 优先保留动作核心及必要宾语，如“抢购物资”“哄抬米价”“发送定位”“紧急撤离”。
3. 删除主体、时间、地点、灾种名、数量和行为结果，除非删掉后会改变动作含义。
4. 一句含多个独立行为时分成多个 span；禁止为了凑长度把不连续文字拼成一个 span。
5. 不标注“返程、改签、做饭、远程上课”等普通活动，除非文本明确表达避险、求助、救援、管理、
   信息传播或扰序功能。

## 主要依据

- Stowe et al. (2016), *Identifying and Categorizing Disaster-Related Tweets*, ACL SocialNLP.
  https://aclanthology.org/W16-6201/
- Alam et al. (2021), *CrisisBench*, ICWSM.
  https://doi.org/10.1609/icwsm.v15i1.18115
- Young et al. (2020), *A Review of Social Media Use During Disaster Response and Recovery Phases*,
  NIST Technical Note 2086. https://doi.org/10.6028/NIST.TN.2086
- Pavlopoulos et al. (2022), *From the Detection of Toxic Spans in Online Discussions to the Analysis of
  Toxic-to-Civil Transfer*, ACL. https://doi.org/10.18653/v1/2022.acl-long.259

