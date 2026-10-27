# disaster-emotion-dbn

**微博帖子 → 情绪–行为时空动力学**：面向中国四类自然灾害（地震、台风、
滑坡、泥石流）的三阶段研究管线，按论文版本分为 v1 / v2 / v3 三套独立实现。
三个版本的 Stage 1 完全共用。

```
Stage 1  微博情绪分类（chinese-roberta-wwm-ext，SMP2020-EWECT，6 类）
             │  为每条帖子输出 EMO（argmax 标签）与 EMO_probs（6 类概率）
             ▼
Stage 2  行为/危害 span 抽取（RoBERTa）
             │  为每条帖子输出行为概率 + 行为文本片段
             ▼
Stage 3  时空动态贝叶斯网络（R + Stan，ADVI）
             事件周围 K 距离环 x 20 个 12 小时时间窗；AR(1) 情绪/行为
             潜变量，同期（θ₁）与滞后（θ₂）耦合；台风嵌套结构；
             环级调节变量（人口/GDP）
```

## 目录结构

```
├── stage1-sentiment/     # 三个版本共用的情绪分类器
├── data/                 # Stage 1 输出：四个灾种帖子级 RDS + eventlist.csv
├── data_stage2/          # v3 的 Stage 2 预测输出（Stage 3 输入）
├── docs/                 # 论文与数据归档材料
├── v1/                   # 第一版：情绪门控危害 span 抽取
│   ├── stage2-span-extraction/   # 情绪概率参与抽取（门控 + 概率校准）
│   └── stage3-dbn/               # 逐事件 CSV 输入（weibo_dbn/input）
├── v2/                   # 第二版：纯文本行为抽取，脱离情绪概率
│   ├── stage2-behavior-extraction/   # 只读文本的行为概率 + 多 span
│   └── stage3-dbn/                   # 输入换为 v2 行为概率（input_v2_v1layout）
└── v3/                   # 第三版：新训练集 + 独立接口 + 模型修正
    ├── stage2-behavior-extraction/   # 1000 条训练集，CRF 多 span
    └── stage3-dbn/                   # 读 data_stage2 RDS；Haversine 距离；
                                     # 情绪缺失按缺失值处理（不补零）
```

## 三个版本的差异

| | v1 | v2 | v3 |
|---|---|---|---|
| Stage 2 模型 | RoBERTa 情绪门控 span 抽取（start/end + 二分类），危害概率用 Stage 1 情绪分布做熵校准 | 纯文本行为抽取（span + 存在概率），不使用任何情绪输入 | 同源代码，训练集更换为 1000 条多来源标注集（CRF 多 span） |
| Stage 2 → Stage 3 接口 | 逐事件 CSV（`raw_prob`） | 逐事件 CSV（`raw_prob`，来自行为模型） | 仓库级 RDS（`data_stage2/*.rds`，`behavior_prob` 字段） |
| Stage 3 输入 | `weibo_dbn/input` | `weibo_dbn/input_v2_v1layout` | `data_stage2/` |
| 距离计算 | 平面近似（经纬度缩放） | 同 v1 | Haversine 球面距离 |
| 情绪缺失值 | 解析失败按 0（中性）处理 | 同 v1 | 无效/缺失记为缺失值（NA），对应格子该通道不进入似然 |
| 观测权重 | 情绪与行为共用格子帖子数 N（1/√N 与 φ·N） | 同 v1 | 同 v1（共用 N；情绪/行为的有效观测由缺失标记区分） |

## 运行环境

- Python 依赖见各目录 `requirements.txt`；PyTorch 在本机（Apple Silicon）
  默认走 MPS，无需 CUDA。stage2 可用 `v3/stage2-behavior-extraction/.venv`。
- R 依赖锁定于各 `stage3-dbn/renv.lock`（tidyverse、cmdstanr、data.table、
  terra/sf/exactextractr 等），另需 [CmdStan](https://mc-stan.org/cmdstan/)。
- 预训练底座与外部数据（本机归档，不入库）：
  - `毕设数据/Thesis-ZJU/span_extract/resource/chinese-roberta-wwm-ext`
    （hfl/chinese-roberta-wwm-ext 本地副本）
  - `毕设数据/Thesis-ZJU/SA_NER/`（SMP2020 训练/验证文本与训练好的
    Stage 1 权重 `checkpoints/best_cls_1.pt`）
  - `毕设数据/Thesis-ZJU/weibo_dbn/`（v1/v2 的逐事件 CSV 输入）
  - `毕设数据/Thesis-ZJU/DBNs/`（event_features.csv、人口/GDP 栅格、
    历史输出存档）

## 快速开始

```bash
# Stage 1（共用）：对 data/ 中 CSV/RDS 打情绪标签
python stage1-sentiment/predict.py --input <CSV/RDS 或目录>

# v3 全流程
python v3/stage2-behavior-extraction/train.py --mode final
python v3/stage2-behavior-extraction/predict.py
Rscript v3/stage3-dbn/runs/build_spatiotemporal_data.R
Rscript v3/stage3-dbn/runs/rerun_all_from_raw.R
Rscript v3/stage3-dbn/runs/rerun_final_30k.R
python v3/stage3-dbn/runs/make_figures.py

# v1 / v2 的 Stage 3 复现
Rscript v1/stage3-dbn/runs/rerun_all_from_raw.R
Rscript v2/stage3-dbn/runs/rerun_full_pipeline.R
```

## 数据与第三方输入（不再分发）

| 输入 | 用途 | 来源 |
|---|---|---|
| SMP2020-EWECT `usual_train.txt` / `usual_eval_labeled.txt` | Stage 1 训练/验证 | SMP2020 挑战赛 |
| `hfl/chinese-roberta-wwm-ext` | 两个 NLP 阶段的底座 | Hugging Face |
| v1 span 训练 CSV（`text`/`span`/`EMO_probs`/`NONE`） | v1 Stage 2 训练 | 人工标注 |
| v2/v3 行为标注 CSV | v2/v3 Stage 2 训练 | 人工标注 + 检索补充 |
| `data/<hazard>.rds` | Stage 1 输出、Stage 2 输入 | 采集微博数据 |
| `data/eventlist.csv` | 事件协变量与中心（震级/深度、降雨、台风等级） | 逐事件整理 |
| 2020 中国 1km 人口栅格 / GDP 栅格 | 环级调节变量与地图 | WorldPop / gridded GDP |

## 复现性

- 全部随机种子固定（PyTorch 42 / ADVI 123 / 图表模拟 123）。
- `renv.lock` 锁定 R 环境；各阶段 `requirements.txt` 列出 Python 依赖。
- 台风去重：同一台风内按最近路径点归属去重，跨台风重叠保留；
  确定性逐行解析器保证不静默截断（data.table::fread 会在未转义引号行截断）。
