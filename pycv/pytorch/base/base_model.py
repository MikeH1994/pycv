from __future__ import annotations
import torch.nn as nn
import torch


class BaseModel(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def process_predictions(self, x: torch.FloatTensor):
        return x