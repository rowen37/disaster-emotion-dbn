"""
Stage 2 - emotion-gated harm-span extractor: model definition.

RoBERTa encoder with an emotion gate: the 6-class emotion probabilities from
Stage 1 are mapped to a per-dimension sigmoid gate that modulates the token
representations before the span heads. Heads: span start, span end, and a
binary "contains harm description" classifier on [CLS].
"""
import torch
import torch.nn as nn
from transformers import AutoModel

class RoBERTaGateSpanExtractor(nn.Module):
    def __init__(self, model_name_or_path, hidden_size=768, emo_dim=6, dropout=0.1):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name_or_path)
        self.hidden_size = hidden_size

        self.emotion_gate = nn.Sequential(
            nn.Linear(emo_dim, hidden_size),
            nn.Sigmoid()
        )
        self.start_fc = nn.Linear(hidden_size, 1)
        self.end_fc = nn.Linear(hidden_size, 1)
        self.has_harm_fc = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, emo_probs):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]

        gate = self.emotion_gate(emo_probs).unsqueeze(1)
        gated_output = sequence_output * gate
        gated_output = self.dropout(gated_output)

        start_logits = self.start_fc(gated_output).squeeze(-1)
        end_logits = self.end_fc(gated_output).squeeze(-1)
        has_harm_logits = self.has_harm_fc(cls_output)

        return start_logits, end_logits, has_harm_logits
