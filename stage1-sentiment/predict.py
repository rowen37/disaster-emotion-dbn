"""
Stage 1 - Weibo emotion classification: batch prediction.

Loads the fine-tuned checkpoint and annotates a CSV of Weibo posts with two
new columns: EMO (argmax label) and EMO_probs (full 6-class probability
vector, JSON-encoded). EMO_probs is the input consumed by Stage 2 (span
extraction) and Stage 3 (DBN gridding).
"""
import csv
import torch
import json
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_PATH = "/Users/rogerbqw/Downloads/span_extract/resource/chinese-roberta-wwm-ext"
MODEL_WEIGHT = "checkpoints/best_cls_1.pt"

INPUT_PATH = "/Users/rogerbqw/Desktop/cleaned_cleaned_补抓专用.csv"
OUTPUT_PATH = "./402_plus_cls.csv"

MAX_LEN = 64
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

label2id = {
    "fear": 0,
    "neutral": 1,
    "sad": 2,
    "surprise": 3,
    "angry": 4,
    "happy": 5
}
id2label = {v: k for k, v in label2id.items()}

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=6
)
model.load_state_dict(torch.load(MODEL_WEIGHT, map_location=DEVICE))
model.to(DEVICE)
model.eval()

import json
import torch.nn.functional as F

def predict_with_probs(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # shape (6,)
        pred_id = probs.argmax()
        pred_label = id2label[pred_id]

    prob_dict = {id2label[i]: float(probs[i]) for i in range(len(id2label))}
    return pred_label, prob_dict

def process():
    with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
         open(OUTPUT_PATH, "w", encoding="utf-8", newline='') as fout:

        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames + ["EMO", "EMO_probs"]   # append the two new columns
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(reader, desc="Predicting emotions"):
            text = row["text"]
            emo_label, prob_dict = predict_with_probs(text)
            row["EMO"] = emo_label
            row["EMO_probs"] = json.dumps(prob_dict, ensure_ascii=False)
            writer.writerow(row)

if __name__ == "__main__":
    process()
