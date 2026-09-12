"""
Stage 2 - emotion-gated harm-span extraction: dataset and label alignment.

Converts annotated posts (text, span, EMO_probs) into training examples:
char-level spans are aligned to token positions via offset mapping, the
Stage 1 emotion vector is attached to every example, and negative posts
(span = NONE) can be merged in at train time. Includes random token-masking
augmentation.
"""
import json
import torch
import random
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from config import Config

cfg = Config()

def parse_emotion_probs(emo_probs_str):
    """EMO_probs column (JSON string or dict) -> fixed-order 6-dim tensor."""
    if isinstance(emo_probs_str, str):
        probs = json.loads(emo_probs_str)
    else:
        probs = emo_probs_str
    return torch.tensor([probs.get(k, 0.0) for k in cfg.emotion_order], dtype=torch.float)

def align_span_to_tokens(text, span_str, tokenizer, max_len):
    """Locate a char-level span in the tokenized text.

    Returns (start_token, end_token, valid); invalid spans (not found in
    text, or lost to truncation) are mapped to the ignore index -100.
    """
    if span_str == "NONE" or pd.isna(span_str) or span_str == "":
        return -100, -100, False

    start_char = text.find(span_str)
    if start_char == -1:
        span_stripped = span_str.strip()
        start_char = text.find(span_stripped)
        if start_char == -1:
            print(f"Warning: cannot find span '{span_str}' in text: {text[:50]}...")
            return -100, -100, False
        else:
            span_str = span_stripped
    end_char = start_char + len(span_str)

    encoding = tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        padding=False,
        return_offsets_mapping=True,
        return_tensors=None
    )
    offsets = encoding['offset_mapping']
    input_ids = encoding['input_ids']

    start_token = None
    end_token = None
    for i, (offset_start, offset_end) in enumerate(offsets):
        if offset_start <= start_char < offset_end:
            start_token = i
        if offset_start < end_char <= offset_end:
            end_token = i
            break

    if start_token is None or end_token is None:
        print(f"Warning: span out of range after truncation: '{span_str}' in text: {text[:50]}...")
        return -100, -100, False

    if start_token >= max_len - 1 or end_token >= max_len - 1:
        return -100, -100, False
    return start_token, end_token, True

class HarmSpanDataset(Dataset):
    """Training/inference dataset for the harm-span extractor.

    Training rows require columns: text, span ('NONE' for no harm),
    EMO_probs. Negative rows (forced span='NONE', neutral emotion) can be
    merged in via negative_df.
    """

    def __init__(self, df, tokenizer, max_len=64, is_train=True, negative_df=None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train
        self.examples = []

        if is_train and negative_df is not None:
            negative_df = negative_df.copy()
            negative_df['span'] = 'NONE'
            negative_df['EMO'] = 'neutral'
            if 'EMO_probs' not in negative_df.columns:
                negative_df['EMO_probs'] = '{"fear":0.0,"neutral":1.0,"sad":0.0,"surprise":0.0,"angry":0.0,"happy":0.0}'
            df = pd.concat([df, negative_df], ignore_index=True)
            print(f"After merging negative samples: total training samples = {len(df)}")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
            text = str(row['text']).strip()
            if not text:
                continue
            emo_probs = parse_emotion_probs(row['EMO_probs'])

            if is_train:
                span_str = str(row.get('span', 'NONE'))
                start_token, end_token, valid = align_span_to_tokens(text, span_str, tokenizer, max_len)
                if not valid and span_str != "NONE":
                    continue
                has_harm = 1 if (span_str != "NONE" and valid) else 0
                if not valid:
                    start_token, end_token = -100, -100
                self.examples.append({
                    'text': text,
                    'emo_probs': emo_probs,
                    'start_label': start_token,
                    'end_label': end_token,
                    'has_harm_label': has_harm
                })
            else:
                self.examples.append({
                    'text': text,
                    'emo_probs': emo_probs,
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # augmentation: randomly mask inner tokens (training only)
        if self.is_train and random.random() < 0.5:
            mask_token_id = self.tokenizer.mask_token_id
            input_ids_aug = input_ids.clone()
            seq_len = attention_mask.sum().item()
            for pos in range(1, seq_len - 1):
                if random.random() < cfg.augment_mask_prob:
                    input_ids_aug[pos] = mask_token_id
            input_ids = input_ids_aug

        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'emo_probs': item['emo_probs']
        }
        if self.is_train:
            result['start_label'] = torch.tensor(item['start_label'], dtype=torch.long)
            result['end_label'] = torch.tensor(item['end_label'], dtype=torch.long)
            result['has_harm_label'] = torch.tensor(item['has_harm_label'], dtype=torch.long)
        else:
            result['text'] = item['text']
        return result
