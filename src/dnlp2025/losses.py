import torch
import torch.nn as nn
from torch.nn.functional import log_softmax


class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, vocab_size, ignore_index=-100):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum') 
        self.label_smoothing = label_smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred = pred.view(-1, self.vocab_size)
        target = target.view(-1)

        mask = target != self.ignore_index
        pred = pred[mask]
        target = target[mask]

        if pred.size(0) == 0:  # Edge case: nothing to compute
            return torch.tensor(0.0, requires_grad=True, device=pred.device)

        with torch.no_grad():
            true_dist = torch.full_like(pred, self.label_smoothing / (self.vocab_size - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)

        # Normalize by number of non-padding tokens
        num_tokens = pred.size(0)
        loss = self.criterion(log_softmax(pred, dim=-1), true_dist)
        
        return loss / num_tokens
