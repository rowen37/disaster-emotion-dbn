"""
Stage 1 - auxiliary: SMP2020-EWECT training-set class distribution.

Parses the usual_train.txt JSON array (tolerating trailing commas), counts
samples per emotion class, and renders the class-distribution bar chart.
"""
import json
import matplotlib.pyplot as plt
import re
from collections import Counter

file_path = 'smp2020_data/usual_train.txt'

# ---------- load and parse the JSON array ----------
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()
content = re.sub(r',\s*\]', ']', content)   # strip trailing commas
data_list = json.loads(content)
labels = [item['label'] for item in data_list]

emotion_order = ['fear', 'neutral', 'sad', 'surprise', 'angry', 'happy']
counts = [labels.count(emo) for emo in emotion_order]

print("各类别样本数量：")
for emo, cnt in zip(emotion_order, counts):
    print(f"{emo:10s}: {cnt}")

# ---------- bar chart ----------
plt.figure(figsize=(12, 4.5))

colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f']

bars = plt.bar(emotion_order, counts, color=colors, edgecolor='black', alpha=0.85)

# annotate counts on top of each bar
for bar, cnt in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
             f'{cnt}', ha='center', va='bottom', fontsize=10)

plt.ylabel('Number of Samples', fontsize=12)
plt.title('Class Distribution of SMP2020 EWECT Training Set', fontsize=14)
plt.xticks(rotation=45, fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
