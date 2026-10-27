"""
Stage 1 - Weibo emotion classification: validation diagnostics.

Runs the fine-tuned classifier over the SMP2020 validation split and
produces (1) the confusion matrix heatmap and (2) per-example probability
bar charts for the first N posts. The aggregate metrics live in
eval_metrics.json next to this file.
"""
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']   # CJK-capable font on macOS
plt.rcParams['axes.unicode_minus'] = False

MODEL_PATH = "/Users/rogerbqw/Downloads/毕设数据/Thesis-ZJU/span_extract/resource/chinese-roberta-wwm-ext"
VAL_PATH = "/Users/rogerbqw/Downloads/毕设数据/Thesis-ZJU/SA_NER/smp2020_data/usual_eval_labeled.txt"
SAVE_PATH = "/Users/rogerbqw/Downloads/毕设数据/Thesis-ZJU/SA_NER/checkpoints/best_cls_1.pt"

MAX_LEN = 64
BATCH_SIZE = 16
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

from transformers import BertTokenizer
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    def __init__(self, path, tokenizer):
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["content"]
        label = label2id[item["label"]]

        encoding = self.tokenizer(
            text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label),
            "text": text
        }

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=6
)
model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------- inference: collect predictions and full probability vectors ----------
val_dataset = EmotionDataset(VAL_PATH, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

all_texts = []
all_true = []
all_pred = []
all_probs = []   # 6 probabilities per post

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].cpu().numpy()
        texts = batch["text"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_texts.extend(texts)
        all_true.extend(labels)
        all_pred.extend(preds)
        all_probs.extend(probs)

cm = confusion_matrix(all_true, all_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(id2label.values()),
            yticklabels=list(id2label.values()))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Emotion Classification')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()
print("混淆矩阵已保存为 confusion_matrix.png")

num_examples = 5 # 5
emotion_names = list(id2label.values())

for i in range(min(num_examples, len(all_texts))):
    text = all_texts[i]
    true_label = id2label[all_true[i]]
    pred_label = id2label[all_pred[i]]
    probs = all_probs[i]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(emotion_names, probs, color='skyblue')

    pred_idx = all_pred[i]
    bars[pred_idx].set_color('orange')
    plt.ylim(0, 1)
    plt.ylabel('Probability')
    plt.title(f'Example {i+1}\nTrue: {true_label} | Pred: {pred_label}\nText: {text[:50]}...')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'prob_distribution_example_{i+1}.png', dpi=300)
    plt.show()

print("概率分布图已保存为 prob_distribution_example_*.png")
