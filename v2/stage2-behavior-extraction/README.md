# Stage 2：行为存在概率与多 span 抽取（v2）

训练入口只使用 `data/behavior_training.csv`（原始人工标注集中通过一致性
校验的子集）。模型只读帖子文本，输出：

- `behavior_prob`：帖子包含目标行为的概率（预测同时写入 `raw_prob`，即
  Stage 3 读取的行为通道字段）。
- `behavior_spans`：行为的最小连续文本片段；一条帖子允许 0 到多个 span。

`behavior_types` 是人工分析字段，不作为模型预测目标。模型不读取 Stage 1
的情绪字段，不训练 consequence 或 severity 头。

## 目录内容

- 代码：`prepare_data.py`（标注校验 + 防事件泄漏五折）、`train.py`
  （`--check-only` / `--mode cv` / `--mode final`）、`evaluate.py`、
  `predict.py` / `dedup_predict.py`（CSV/RDS 推理）、`validate_spans.py`。
- `data/behavior_training.csv`：训练集。
- `checkpoints/` 与 `runs/`：本版训练模型与交叉验证评估的存档
  （联合架构 span+presence 模型，二分类 F1≈0.820，建议部署阈值 0.34）。
  存档检查点的网络结构与当前代码（CRF 头）不同，不能被 `predict.py`
  直接加载；如需重新推理请先重训。

## 使用

```bash
python prepare_data.py
python train.py --check-only
python train.py --mode cv
python evaluate.py
python train.py --mode final
python predict.py
```
