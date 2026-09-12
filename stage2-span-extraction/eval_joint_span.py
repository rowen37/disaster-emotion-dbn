# =====================================================================
# Stage 2 - joint evaluation of the archived best_model.pth on the same
# validation split used during training. Reports:
#   1) the legacy conditional metric (independent argmax on gold-span
#      posts) for reconciliation with the training log (0.803)
#   2) a clean re-evaluation (no random-mask augmentation, which leaked
#      into validation during training via the is_train branch)
#   3) joint strict exact-match: a span counts only when the binary head
#      fires; a predicted span on a NONE post is an FP, an imprecise
#      boundary on a positive post is an FP+FN
#   4) SQuAD-style character-overlap F1 (micro / macro)
# Outputs: eval_joint_results.json + eval_joint_details.csv
# =====================================================================
import json
import os
import sys
from collections import Counter

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from model import RoBERTaGateSpanExtractor
from data_utils import align_span_to_tokens, parse_emotion_probs

cfg = Config()
torch.manual_seed(0)

# ---------- 1) rebuild train.py's validation split and filtering (no mask augmentation) ----------
df = pd.read_csv(cfg.train_file, encoding="utf-8")
df = df.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
val_df = df.iloc[int(0.9 * len(df)):]

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)

examples = []
for _, row in val_df.iterrows():
    text = str(row["text"]).strip()
    if not text:
        continue
    span_str = str(row.get("span", "NONE"))
    s_tok, e_tok, valid = align_span_to_tokens(text, span_str, tokenizer, cfg.max_len)
    if not valid and span_str != "NONE":
        continue
    has_harm = 1 if (span_str != "NONE" and valid) else 0
    if not valid:
        s_tok, e_tok = -100, -100
    examples.append(dict(text=text, gold_span=(span_str if has_harm else None),
                         s=s_tok, e=e_tok, y=has_harm,
                         emo=parse_emotion_probs(row["EMO_probs"])))
print(f"有效验证集: {len(examples)} 帖 "
      f"(带 span {sum(e['y'] for e in examples)}, NONE {sum(1 - e['y'] for e in examples)})")

# ---------- 2) load the archived checkpoint ----------
model = RoBERTaGateSpanExtractor(cfg.model_name_or_path)
sd = torch.load(os.path.join(cfg.checkpoint_dir, "best_model.pth"),
                map_location="cpu", weights_only=True)
model.load_state_dict(sd)
model.eval()

def char_f1(gold, pred):
    cg, cp = Counter(gold), Counter(pred)
    common = sum((cg & cp).values())
    if common == 0:
        return 0.0, 0, sum(cg.values()), [sum(cp.values()) if False else sum(cp.values())][0]
    p = common / max([sum(cp.values()) if False else sum(cp.values())][0], 1)
    r = common / max(sum(cg.values()), 1)
    return 2 * p * r / (p + r), common, sum(cg.values()), [sum(cp.values()) if False else sum(cp.values())][0]

details = []
with torch.no_grad():
    for ex in examples:
        enc = tokenizer(ex["text"], max_length=cfg.max_len,
                        padding="max_length", truncation=True, return_tensors="pt")
        s_logit, e_logit, h_logit = model(enc["input_ids"], enc["attention_mask"],
                                          ex["emo"].unsqueeze(0))
        y_hat = int(h_logit.argmax())

        # legacy metric: independent argmax, same construction as compute_span_f1
        old_s, old_e = int(s_logit.argmax()), int(e_logit.argmax())

        # clean decoding: (s, e) with maximal logit sum, s <= e, within valid length
        L = int(enc["attention_mask"].sum())
        score = s_logit[0].unsqueeze(1) + e_logit[0].unsqueeze(0)
        sc = torch.full_like(score, -1e9)
        allow = torch.triu(torch.ones(L, L, dtype=torch.bool))
        sc[:L, :L][allow] = score[:L, :L][allow]
        idx = int(sc.argmax())
        s_hat, e_hat = idx // sc.shape[1], idx % sc.shape[1]
        pred_span = tokenizer.decode(enc["input_ids"][0][s_hat:e_hat + 1],
                                     skip_special_tokens=True).strip()
        details.append(dict(text=ex["text"][:60], y=ex["y"], y_hat=y_hat,
                            gold_span=ex["gold_span"],
                            pred_span=pred_span if y_hat == 1 else None,
                            em=int(y_hat == ex["y"] and ((ex["y"] == 0) or
                                (s_hat == ex["s"] and e_hat == ex["e"]))),
                            old_em=int(ex["y"] == 1 and old_s == ex["s"] and old_e == ex["e"])))

d = pd.DataFrame(details)

# ---------- 3) metrics ----------
def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)

# binary head (clean re-evaluation)
y, yh = d.y.values, d.y_hat.values
tp = int(((y == 1) & (yh == 1)).sum()); fp = int(((y == 0) & (yh == 1)).sum())
fn = int(((y == 1) & (yh == 0)).sum()); tn = int(((y == 0) & (yh == 0)).sum())
binary = dict(acc=round((tp + tn) / len(d), 4), **dict(zip(("P", "R", "F1"), prf(tp, fp, fn))))

# legacy metric: exact boundary hit rate on gold-span posts (independent argmax)
gold_n = int((d.y == 1).sum())
old_em_rate = round(d.loc[d.y == 1, "old_em"].mean(), 4)

# joint strict EM: a span is predicted only when y_hat = 1
tp_j = int(((d.y == 1) & (d.y_hat == 1) & (d.em == 1)).sum())
fp_j = int(((d.y_hat == 1) & (d.em == 0)).sum())          # span on NONE post + imprecise boundary
fn_j = int(((d.y == 1) & ((d.y_hat == 0) | (d.em == 0))).sum())
joint_em = dict(TP=tp_j, FP=fp_j, FN=fn_j,
                none_fp=int(((d.y == 0) & (d.y_hat == 1)).sum()),
                **dict(zip(("P", "R", "F1"), prf(tp_j, fp_j, fn_j))))

# character overlap (SQuAD-style): corpus micro + per-post macro (NONE with no prediction counts as 1)
tp_c = fp_c = fn_c = 0
per_post = []
for _, r in d.iterrows():
    gold = r.gold_span if isinstance(r.gold_span, str) else ""
    pred = r.pred_span if isinstance(r.pred_span, str) else ""
    f1c, common, ng, np_ = char_f1(gold, pred)
    tp_c += common; fp_c += np_ - common; fn_c += ng - common
    per_post.append(1.0 if (not gold and not pred) else f1c)
overlap_micro = dict(zip(("P", "R", "F1"), prf(tp_c, fp_c, fn_c)))
overlap_macro = round(sum(per_post) / len(per_post), 4)

out = dict(checkpoint="best_model.pth (run 38zzvvox, epoch 5, 2026-04-16 06:21)",
           val_effective_n=len(d), val_gold_span=gold_n,
           binary=binary, conditional_EM_old_metric=old_em_rate,
           logged_in_paper=dict(acc=0.9612, F1=0.9791, span_f1=0.8034),
           joint_exact_match=joint_em,
           char_overlap_micro=overlap_micro, char_overlap_macro_all=overlap_macro)
print(json.dumps(out, ensure_ascii=False, indent=2))
d.to_csv("eval_joint_details.csv", index=False)
with open("eval_joint_results.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("已保存 eval_joint_results.json / eval_joint_details.csv")
