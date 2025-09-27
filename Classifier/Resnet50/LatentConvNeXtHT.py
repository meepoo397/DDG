import torch
import torch.nn as nn

# ------------------ 基本模組 ------------------
class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_value)
    def forward(self, x): return x * self.gamma.view(1, -1, 1, 1)

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, ls_init=1e-6):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.ls = LayerScale(dim, ls_init)
    def forward(self, x):
        shortcut = x
        x = self.dw(x)
        x = x.permute(0, 2, 3, 1)      # to channels-last
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)      # back to NCHW
        x = self.pw2(self.act(self.pw1(x)))
        x = self.ls(x)
        return x + shortcut

class Downsample(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pre = nn.Conv2d(in_c, in_c, 3, stride=1, padding=1, groups=in_c)
        self.down = nn.Conv2d(in_c, out_c, 2, stride=2)
    def forward(self, x): return self.down(self.pre(x))

class LatentAdapter(nn.Module):
    def __init__(self, in_c=4, out_c=16):
        super().__init__()
        self.proj = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.norm = nn.LayerNorm(out_c, eps=1e-6)
    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)

class TransformerHead(nn.Module):
    def __init__(self, dim, num_tokens, depth=4, heads=16, mlp_ratio=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim*mlp_ratio,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, num_tokens+1, dim))
    def forward(self, x):  # x: B,C,H,W
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, HW, C
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        x = self.enc(x)
        return x[:, 0]  # CLS token

# ------------------ 主架構 ------------------
class LatentConvNeXtHT(nn.Module):
    def __init__(self, num_classes=1000, in_channels=4,
                 dims=(128, 256, 512, 1024), depths=(3, 3, 27, 3),
                 adapter_channels=16, transformer_depth=4, transformer_heads=16,
                 dropout_head=(0.5, 0.3)):
        super().__init__()
        self.adapter = LatentAdapter(in_channels, adapter_channels)
        c1 = dims[0]
        self.stem = nn.Conv2d(adapter_channels, c1, kernel_size=2, stride=2)  # 64→32

        stages = []
        in_dim = c1
        for i, (d, dim) in enumerate(zip(depths, dims)):
            if i > 0:
                stages.append(Downsample(in_dim, dim))
            blocks = [ConvNeXtBlock(dim) for _ in range(d)]
            stages += blocks
            in_dim = dim
        self.stages = nn.Sequential(*stages)

        # 最終解析度 4×4 → token=16
        self.trans = TransformerHead(in_dim, num_tokens=16,
                                     depth=transformer_depth, heads=transformer_heads)

        # 改進的分類頭（正則化）
        self.head = nn.Sequential(
            nn.Dropout(dropout_head[0]),
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_head[1]),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.adapter(x)
        x = self.stem(x)
        x = self.stages(x)
        x = self.trans(x)
        return self.head(x)

# ------------------ 工廠函數 ------------------
def create_latent_convnext_ht_classifier(num_classes=1000,
                                         in_channels=4,
                                         pretrained=False,
                                         variant="base"):
    """
    建立適用 VAE latent 空間的 ConvNeXt + Transformer 混合分類器

    Args:
        num_classes (int): 分類類別數
        in_channels (int): 輸入通道數 (VAE latent 預設 4)
        pretrained (bool): 是否載入預訓練 (目前僅初始化，不支援直接載入 ImageNet RGB 權重)
        variant (str): "small" | "base" | "large"
    """
    if variant == "small":
        dims = (96, 192, 384, 768)
        depths = (2, 2, 12, 2)
        t_depth, t_heads = 2, 8
    elif variant == "large":
        dims = (192, 384, 768, 1536)
        depths = (3, 3, 27, 3)
        t_depth, t_heads = 6, 24
    else:  # base
        dims = (128, 256, 512, 1024)
        depths = (3, 3, 27, 3)
        t_depth, t_heads = 4, 16

    model = LatentConvNeXtHT(
        num_classes=num_classes,
        in_channels=in_channels,
        dims=dims,
        depths=depths,
        adapter_channels=16,
        transformer_depth=t_depth,
        transformer_heads=t_heads
    )

    # 權重初始化
    if not pretrained:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    return model
