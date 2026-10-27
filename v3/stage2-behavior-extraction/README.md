# Stage 2：有害相关信号评分与多 span 抽取（v3）

训练入口只使用 `data/behavior_training.csv` 的 1000 条样本。所有样本处于同一训练层级，不按原始、检索补充或人工编写来源区别加权。

统一提取目标是：可能伤害本人、他人、公共秩序或资源分配的相关信号，包括实际有害行为、明确的有害主体身份、失控的即时反应和直接有害后果。例如违法失范、谣言诈骗、哄抬盗抢、破坏阻碍、恐慌性抢购囤积、明显冒险暴露，以及“骗子”“爬起来就跑”“引发大量群众恐慌”等与上述风险直接相关的文本证据。正常且有计划的避险、求助、救援、信息报告和应急管理仍为负例；但同一句中若另有失控反应或有害后果，相关片段应单独标注。

模型只读帖子文本，输出：

- `behavior_prob`：连续的广义有害相关信号分数，供排序和 Stage 3 使用，不负责删除 span。
- `behavior_spans`：CRF 独立解码出的一个或多个证据片段。
- `has_behavior`：`behavior_spans` 是否非空。

`behavior_types` 统一为 `有害行为`，仅用于标注审计，不作为子类别或不同训练等级。模型不读取 Stage 1 情绪字段，也不读取 volume，不训练 consequence 或 severity 头。

## 使用

```bash
python prepare_data.py             # 只校验标注、分组和五折，不改训练集
python prepare_data.py --rebalance # 仅在折数不等时按完整 group 做最小调整
python train.py --check-only       # 路径、模式、折文件检查，不加载模型
python validate_spans.py           # 可选：tokenizer 覆盖检查
python train.py --mode cv          # 5 折 × 3 种子交叉验证
python evaluate.py                 # 汇总 CV 指标
python train.py --mode final       # 全量训练部署模型
python predict.py                  # 对仓库 data/ 打分，输出至 data_stage2/
```

五折各 200 条；相同事件或重复文本 group 不跨折。训练、评估和推理均不设置二值截断值：presence 头保留为连续分数，span 与 `has_behavior` 完全由 CRF 解码结果决定。

产物写入本目录 `outputs/`（checkpoint、训练日志、CV 指标）与仓库级 `data_stage2/`（预测文件）。Python 依赖见 `requirements.txt`。
