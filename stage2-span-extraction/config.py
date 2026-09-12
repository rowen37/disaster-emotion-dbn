"""
Stage 2 - emotion-gated harm-span extraction: shared configuration.

All paths and hyperparameters for Stage 2 live here. Inputs are Weibo posts
already annotated with EMO_probs by Stage 1; the model extracts the span
describing harmful consequences (or decides there is none) and outputs a
per-post harm probability (raw_prob), which becomes the behavior channel
observed by the Stage 3 DBN.
"""
import os

class Config:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model_name_or_path = os.path.join(base_dir, "/Users/rogerbqw/Downloads/span_extract/resource/chinese-roberta-wwm-ext")
    train_file = os.path.join(base_dir, "/Users/rogerbqw/Downloads/span_extract/data_input/秘制小训练集_augmented.csv")
    checkpoint_dir = os.path.join(base_dir, "/Users/rogerbqw/Downloads/span_extract/checkpoints")
    negative_samples_file = os.path.join(base_dir, "/Users/rogerbqw/Downloads/span_extract/data_input/negative.csv")

    test_file = os.path.join(base_dir, "/Users/rogerbqw/Downloads/SA_NER/cls_output_typhoon.csv/408_cls.csv")
    output_file = os.path.join(base_dir, "/Users/rogerbqw/Downloads/sepclsspan/408.csv")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    max_len = 64
    batch_size = 8
    epochs = 5
    lr = 1e-4
    warmup_ratio = 0.1
    weight_decay = 0.01
    seed = 42
    device = "cuda"

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
