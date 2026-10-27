"""
Stage 1 - Weibo emotion classification: training.

Fine-tunes chinese-roberta-wwm-ext on the SMP2020-EWECT dataset for 6-way
emotion classification (fear / neutral / sad / surprise / angry / happy).
The best checkpoint by validation macro-F1 is saved for batch prediction.
"""
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import wandb

MODEL_PATH = "/Users/rogerbqw/Downloads/毕设数据/Thesis-ZJU/span_extract/resource/chinese-roberta-wwm-ext"
TRAIN_PATH = "/Users/rogerbqw/Downloads/毕设数据/Thesis-ZJU/SA_NER/smp2020_data/usual_train.txt"
VAL_PATH = "/Users/rogerbqw/Downloads/毕设数据/Thesis-ZJU/SA_NER/smp2020_data/usual_eval_labeled.txt"
SAVE_PATH = "checkpoints/best_cls_1.pt"
os.makedirs(os.path.dirname(SAVE_PATH) or ".", exist_ok=True)

MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

wandb.init(
    project="roberta_smp2020",
    config={
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "max_len": MAX_LEN
    }
)

label2id = {
    "fear": 0,
    "neutral": 1,
    "sad": 2,
    "surprise": 3,
    "angry": 4,
    "happy": 5
}
id2label = {v: k for k, v in label2id.items()}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "mps":
        torch.mps.manual_seed(seed)

set_seed()

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
            "labels": torch.tensor(label)
        }

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

model = BertForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=6
)

model.to(DEVICE)

train_dataset = EmotionDataset(TRAIN_PATH, tokenizer)
val_dataset = EmotionDataset(VAL_PATH, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

optimizer = AdamW(model.parameters(), lr=LR)

def evaluate(model, loader):
    model.eval()

    preds = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            y = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            pred = torch.argmax(logits, dim=1)

            preds.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro")

    return acc, precision, recall, f1

best_f1 = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for batch in loop:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    acc, p, r, f1 = evaluate(model, val_loader)

    print(f"\nEpoch {epoch+1}")
    print(f"Loss: {total_loss:.4f}")
    print(f"Acc: {acc:.4f} | P: {p:.4f} | R: {r:.4f} | F1: {f1:.4f}")

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": total_loss,
        "val_acc": acc,
        "val_precision": p,
        "val_recall": r,
        "val_f1": f1
    })

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), SAVE_PATH)
        print("Best model saved")

wandb.finish()

print(f"\nBest F1: {best_f1:.4f}")
