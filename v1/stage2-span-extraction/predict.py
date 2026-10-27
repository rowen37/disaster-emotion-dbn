"""
Stage 2 - emotion-gated harm-span extraction: batch prediction.

Loads the best checkpoint, predicts per-post harm probability and decodes
the harm span for posts above the threshold. The raw model probability is
calibrated with the Stage 1 emotion distribution (calibrate_with_entropy)
before thresholding; both raw and calibrated probabilities are written out.
"""
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from config import Config
from model import RoBERTaGateSpanExtractor
from data_utils import HarmSpanDataset

cfg = Config()

def calibrate_with_entropy(original_prob, emo_probs, alpha=None, beta=None):
    """Calibrate the harm probability using the emotion distribution.

    - Dominant negative emotion (max prob > 0.7): small boost proportional
      to concentration.
    - Dominant neutral/happy with high raw probability: small shrink.
    - Otherwise: keep the raw probability.
    Dominant happy (> 0.6) additionally halves the probability (main
    confusion direction of the task).
    """
    if alpha is None:
        alpha = cfg.calibration_alpha
    if beta is None:
        beta = cfg.calibration_beta

    probs = np.array(emo_probs)
    entropy = -np.sum(probs * np.log(probs + 1e-8))
    max_emo = np.max(probs)
    max_idx = np.argmax(probs)
    negative_idx = [0, 2, 4]     # fear, sad, angry

    # concentrated negative emotion: mild boost
    if max_emo > 0.7 and max_idx in negative_idx:
        boost = alpha * (max_emo - 0.7) / 0.3
        prob = original_prob * (1 + boost)
    # concentrated neutral(1)/happy(5): shrink when raw probability is high
    elif max_emo > 0.7 and max_idx in [1, 5]:
        if original_prob > 0.6:
            prob = original_prob * (1 - beta)
        else:
            prob = original_prob
    # high entropy (uncertain emotion): rely fully on the raw probability
    else:
        prob = original_prob

    # dominant happy (> 0.6): force a strong reduction (happy false positives)
    if probs[5] > 0.6:
        prob = prob * 0.5

    return np.clip(prob, 0.0, 1.0)

def predict(model, dataloader, tokenizer, device, threshold=0.5, calibrate=True):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            emo_probs_tensor = batch['emo_probs'].to(device)
            texts = batch['text']

            start_logits, end_logits, has_harm_logits = model(input_ids, attention_mask, emo_probs_tensor)
            has_harm_probs = torch.softmax(has_harm_logits, dim=-1)[:, 1]

            for i, text in enumerate(texts):
                raw_prob = has_harm_probs[i].item()
                emo_vector = emo_probs_tensor[i].cpu().numpy()

                if calibrate:
                    calibrated_prob = calibrate_with_entropy(raw_prob, emo_vector)
                else:
                    calibrated_prob = raw_prob

                has_harm = (calibrated_prob > threshold)

                pred_span = ""
                if has_harm:
                    seq_len = attention_mask[i].sum().item()
                    start_idx = torch.argmax(start_logits[i, 1:seq_len-1]) + 1
                    end_idx = torch.argmax(end_logits[i, 1:seq_len-1]) + 1
                    if end_idx < start_idx:
                        end_idx = start_idx
                    token_ids = input_ids[i][start_idx:end_idx+1].tolist()
                    pred_span = tokenizer.decode(token_ids, skip_special_tokens=True).strip()

                results.append({
                    'text': text,
                    'has_harm': has_harm,
                    'pred_span': pred_span,
                    'raw_prob': raw_prob,
                    'calibrated_prob': calibrated_prob
                })
    return results

def run_predict():
    device = torch.device(
        "mps" if cfg.device == "auto" and torch.backends.mps.is_available() else
        "cuda" if cfg.device == "auto" and torch.cuda.is_available() else
        cfg.device if cfg.device != "auto" else "cpu"
    )
    test_df = pd.read_csv(cfg.test_file, encoding='utf-8')
    if 'EMO_probs' not in test_df.columns:
        raise ValueError("Test file must contain 'EMO_probs' column")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    test_dataset = HarmSpanDataset(test_df, tokenizer, cfg.max_len, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    model = RoBERTaGateSpanExtractor(cfg.model_name_or_path).to(device)
    model_path = os.path.join(cfg.checkpoint_dir, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train first.")
    model.load_state_dict(torch.load(model_path, map_location=device))

    predictions = predict(model, test_loader, tokenizer, device, cfg.has_harm_threshold, calibrate=True)
    out_df = pd.DataFrame(predictions)
    out_df.to_csv(cfg.output_file, index=False, encoding='utf-8-sig')
    print(f"Predictions saved to {cfg.output_file}")

if __name__ == "__main__":
    run_predict()
