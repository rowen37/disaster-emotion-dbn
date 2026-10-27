"""Text-only multi-span behavior extractor."""
import torch
import torch.nn as nn
from transformers import AutoModel


class LinearChainCRF(nn.Module):
    """First-order CRF with hard BIO constraints and batch-first tensors."""

    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.zeros(num_tags))
        self.end_transitions = nn.Parameter(torch.zeros(num_tags))
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))

        # Tag ids are O=0, B=1, I=2.  An I cannot start a sequence or follow O.
        start_constraint = torch.zeros(num_tags)
        start_constraint[2] = -10000.0
        transition_constraint = torch.zeros(num_tags, num_tags)
        transition_constraint[0, 2] = -10000.0
        self.register_buffer("start_constraint", start_constraint)
        self.register_buffer("transition_constraint", transition_constraint)

    def _constrained_parameters(self, start=None, end=None, transitions=None):
        start = self.start_transitions if start is None else start
        end = self.end_transitions if end is None else end
        transitions = self.transitions if transitions is None else transitions
        return start + self.start_constraint, end, transitions + self.transition_constraint

    def neg_log_likelihood(self, emissions, tags, mask):
        """Mean negative log likelihood for contiguous, non-empty masks."""
        mask = mask.bool()
        if emissions.ndim != 3 or tags.shape != mask.shape or emissions.shape[:2] != tags.shape:
            raise ValueError("invalid CRF tensor shapes")
        if not bool(mask[:, 0].all()):
            raise ValueError("every CRF sequence must start with a valid token")
        start, end, transitions = self._constrained_parameters()

        first_tags = tags[:, 0]
        gold_score = start[first_tags] + emissions[:, 0].gather(1, first_tags[:, None]).squeeze(1)
        for position in range(1, emissions.shape[1]):
            active = mask[:, position]
            previous, current = tags[:, position - 1], tags[:, position]
            step = transitions[previous, current]
            step = step + emissions[:, position].gather(1, current[:, None]).squeeze(1)
            gold_score = gold_score + step * active
        last_index = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, last_index[:, None]).squeeze(1)
        gold_score = gold_score + end[last_tags]

        log_partition = start + emissions[:, 0]
        for position in range(1, emissions.shape[1]):
            scores = (
                log_partition.unsqueeze(2)
                + transitions.unsqueeze(0)
                + emissions[:, position].unsqueeze(1)
            )
            next_partition = torch.logsumexp(scores, dim=1)
            log_partition = torch.where(mask[:, position, None], next_partition, log_partition)
        log_partition = torch.logsumexp(log_partition + end, dim=1)
        return (log_partition - gold_score).mean()

    def decode(self, emissions, mask, start=None, end=None, transitions=None):
        """Viterbi decode, returning O-padded tag ids with the input shape."""
        mask = mask.bool()
        if emissions.ndim != 3 or emissions.shape[:2] != mask.shape:
            raise ValueError("invalid CRF decode tensor shapes")
        if not bool(mask[:, 0].all()):
            raise ValueError("every CRF sequence must start with a valid token")
        start, end, transitions = self._constrained_parameters(start, end, transitions)
        score = start + emissions[:, 0]
        history = []
        for position in range(1, emissions.shape[1]):
            candidates = score.unsqueeze(2) + transitions.unsqueeze(0)
            best_score, best_previous = candidates.max(dim=1)
            best_score = best_score + emissions[:, position]
            score = torch.where(mask[:, position, None], best_score, score)
            history.append(best_previous)

        lengths = mask.long().sum(dim=1)
        paths = torch.zeros(mask.shape, dtype=torch.long, device=emissions.device)
        final_tags = (score + end).argmax(dim=1)
        for batch_index, length in enumerate(lengths.tolist()):
            tag = int(final_tags[batch_index])
            decoded = [tag]
            for step in range(length - 2, -1, -1):
                tag = int(history[step][batch_index, tag])
                decoded.append(tag)
            decoded.reverse()
            paths[batch_index, :length] = torch.tensor(decoded, device=paths.device)
        return paths


class BehaviorSpanExtractor(nn.Module):
    """Encoder with one BIO span head and one behavior-presence head."""

    def __init__(self, model_name_or_path, dropout=0.1, n_tags=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.behavior_tag_fc = nn.Linear(hidden_size, n_tags)
        self.behavior_crf = LinearChainCRF(n_tags)
        self.has_behavior_fc = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        sequence = self.dropout(
            self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        )
        cls = sequence[:, 0, :]
        return {
            "behavior_tags": self.behavior_tag_fc(sequence),
            "has_behavior": self.has_behavior_fc(cls),
        }

    def crf_loss(self, emissions, tags, mask):
        return self.behavior_crf.neg_log_likelihood(emissions, tags, mask)

    def crf_decode(self, emissions, mask, ensemble_parameters=None):
        crf = self.behavior_crf
        if ensemble_parameters is None:
            return crf.decode(emissions, mask)
        return crf.decode(emissions, mask, *ensemble_parameters)
