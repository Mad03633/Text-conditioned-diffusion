import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import TIME_EMB_DIM, LABEL_EMB_DIM


def sinusoidal_time_embedding(timesteps, dim):
    half = dim // 2
    factor = math.log(10000) / (half - 1)
    exps = torch.exp(torch.arange(half, device=timesteps.device) * -factor)
    angles = timesteps.float().unsqueeze(1) * exps.unsqueeze(0)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class ConditionalUNet(nn.Module):
    def __init__(self, num_classes=10, time_dim=32, label_dim=32):
        super().__init__()

        self.label_emb = nn.Embedding(num_classes, label_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, 64), nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.label_mlp = nn.Sequential(
            nn.Linear(label_dim, 64), nn.ReLU(),
            nn.Linear(64, 64)
        )

        in_ch = 1 + 64 + 64

        self.down1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
        )

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(384, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(192, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
        )

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x, t, y):
        B, _, H, W = x.size()

        t_emb = self.time_mlp(sinusoidal_time_embedding(t, TIME_EMB_DIM))
        y_emb = self.label_mlp(self.label_emb(y))

        t_map = t_emb.view(B, 64, 1, 1).expand(B, 64, H, W)
        y_map = y_emb.view(B, 64, 1, 1).expand(B, 64, H, W)

        x = torch.cat([x, t_map, y_map], dim=1)

        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))

        mid = self.pool(d2)
        mid = self.bottleneck(mid)

        up = nn.functional.interpolate(mid, scale_factor=2, mode="nearest")
        up = self.up1(torch.cat([up, d2], dim=1))

        up = nn.functional.interpolate(up, scale_factor=2, mode="nearest")
        up = self.up2(torch.cat([up, d1], dim=1))

        return self.out(up)