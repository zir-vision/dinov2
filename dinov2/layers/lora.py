import torch
import torch.nn as nn

from dinov2.layers.block import Attention


class LoRAQKV(nn.Module):
    def __init__(self, attention: Attention, rank: int, alpha: float):
        super(LoRAQKV, self).__init__()
        self.qkv = attention.qkv
        self.scaling = alpha / rank

        self.lora_a = nn.Linear(self.qkv.in_features, rank)
        self.lora_b = nn.Linear(rank, self.qkv.out_features)

    def forward(self, x):
        frozen_qkv = self.qkv(x)
        a = self.lora_a(x)
        b = self.lora_b(a)
        lora_qkv = self.scaling * b
        return frozen_qkv + lora_qkv