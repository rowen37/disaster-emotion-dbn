"""
Stage 2 - emotion-gated harm-span extraction: shared configuration.

All paths and hyperparameters for Stage 2 live here. Inputs are Weibo posts
already annotated with EMO_probs by Stage 1; the model extracts the span
describing harmful consequences (or decides there is none) and outputs a
per-post harm probability (raw_prob), which becomes the behavior channel
observed by the Stage 3 DBN.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class Config:
    base_dir = str(ROOT)

    # Local pre-trained backbone (hfl/chinese-roberta-wwm-ext copy)
    model_name_or_path = "/Users/rogerbqw/Downloads/毕设数据/Thesis-ZJU/span_extract/resource/chinese-roberta-wwm-ext"
    train_file = str(ROOT / "data" / "秘制小训练集_augmented.csv")
    negative_samples_file = str(ROOT / "data" / "negative.csv")
    checkpoint_dir = str(ROOT / "checkpoints")

    # Prediction input: a post CSV annotated with EMO_probs by Stage 1;
    # output is a per-post prediction CSV for Stage 3.
    test_file = str(ROOT / "inputs" / "posts_cls.csv")
    output_file = str(ROOT / "outputs" / "span_predictions.csv")

    max_len = 64
    batch_size = 8
    epochs = 5
    lr = 1e-4
    warmup_ratio = 0.1
    weight_decay = 0.01
    seed = 42
    # auto: mps on Apple Silicon, cuda if present, else cpu
    device = "auto"

    has_harm_threshold = 0.5

    # Emotion-based calibration of the harm probability
    calibration_alpha = 0.2      # boost strength for concentrated negative emotion
    calibration_beta = 0.1       # shrink strength for concentrated neutral/happy

    emotion_order = ['fear', 'neutral', 'sad', 'surprise', 'angry', 'happy']

    # Negative-sample handling
    negative_weight = 1.5        # loss weight for no-harm samples
    use_negative_sampling = True

    # Emotion-aware loss weighting (extra penalty on confident-happy no-harm samples)
    happy_penalty_weight = 2.0

    # On-the-fly augmentation (random token masking)
    augment_mask_prob = 0.1
