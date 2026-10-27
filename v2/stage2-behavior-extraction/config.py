"""Shared configuration for Stage 2 behavior span extraction."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent

# Model
MODEL_NAME = os.environ.get("STAGE2_MODEL", "hfl/chinese-roberta-wwm-ext")
MAX_LEN = 256
DROPOUT = 0.1
N_TAGS = 3  # O / B / I
CHECKPOINT_FORMAT_VERSION = 4
RUN_ID = "behavior_crf_v4"

# Training. Effective batch size = BATCH_SIZE x GRAD_ACCUM_STEPS = 16; the
# micro-batch fits comfortably in a local Apple-Silicon MPS session.
BATCH_SIZE = int(os.environ.get("STAGE2_BATCH_SIZE", "8"))
EVAL_BATCH_SIZE = int(os.environ.get("STAGE2_EVAL_BATCH_SIZE", "16"))
GRAD_ACCUM_STEPS = int(os.environ.get("STAGE2_GRAD_ACCUM_STEPS", "2"))
USE_AMP = os.environ.get("STAGE2_USE_AMP", "0") == "1"
GRADIENT_CHECKPOINTING = os.environ.get("STAGE2_GRADIENT_CHECKPOINTING", "0") == "1"
EPOCHS = 10
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EARLY_STOP_PATIENCE = 3
MASK_AUG_PROB = 0.1
INNER_VAL_FRAC = 0.10
PRESENCE_LOSS_WEIGHT = 0.5
# Cui et al. (2019), "Class-Balanced Loss Based on Effective Number of Samples".
# The CRF NLL is the primary sequence loss; this tempered class-balanced CE is
# auxiliary and avoids raw inverse-frequency B/I over-weighting.
CLASS_BALANCE_BETA = 0.999
TAG_AUX_CE_WEIGHT = 0.25
TAG_WEIGHT_MAX = 5.0

# Evaluation protocol
N_FOLDS = 5
SEEDS = (42, 43, 44)
DEPLOY_SEED = 42
SPLIT_SEED = 42
THRESHOLD = 0.5

# Canonical input and generated split metadata
DATA_TRAIN = ROOT / "data" / "behavior_training.csv"
FOLDS_FILE = ROOT / "data" / "folds.csv"
DATA_REPORT = ROOT / "data" / "data_validation_report.json"
TOKEN_REPORT = ROOT / "data" / "token_validation_report.json"

# All generated artifacts stay inside this module. Prediction output stays in this module's outputs/ by default.
OUTPUT_DIR = ROOT / "outputs"
RUNS_DIR = OUTPUT_DIR / "runs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
REPO_ROOT = ROOT.parents[2]
PREDICT_INPUT = REPO_ROOT / "data"
PREDICT_OUTPUT = OUTPUT_DIR / "predictions"
METRICS_CSV = RUNS_DIR / "metrics.csv"
TRAIN_LOG_CSV = RUNS_DIR / "train_log.csv"
FINAL_INFO = RUNS_DIR / "final_info.json"
