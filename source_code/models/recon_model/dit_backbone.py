import torch
import torch.nn as nn
import math

# ==========================================
# 1. 基础组件：时间步嵌入 (仅保留 t)
# ==========================================
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        ) # [B, frequency_embedding_size] -> [B, hidden_size]
        self.frequency_embedding_size = frequency_embedding_size

        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        args = t[:, None].float() * self.freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        t_emb = self.mlp(embedding)  # mlp for better representation. From (B, F) to (B, hidden_size)
        return t_emb

# ==========================================
# 2. DiT Block (AdaLN 仅受 t 控制)
# ==========================================
class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

        # AdaLN: 现在只接受 t 的 embedding，所以输入维度只是 hidden_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # c 这里仅仅是 t_emb，作为Embedded Time Step (Condition)
        # AdaLN modulation，将c映射为6个向量，并且分割
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        x_norm1 = self.norm1(x) # LayerNorm, x
        x_modulated = x_norm1 * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_modulated, x_modulated, x_modulated)
        x = x + gate_msa.unsqueeze(1) * attn_out

        x_norm2 = self.norm2(x)
        x_modulated = x_norm2 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_modulated)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x

# ==========================================
# 3. 主模型：BatteryDiT
# ==========================================
# 均命名以Model命名以方便调用
class Model(nn.Module):
    def __init__(
        self,
        args,
        input_len=1000,
        in_channels=2,      # [Noisy_V, x_po_input]
        hidden_size=256,    # Default embedding Size
        depth=6,            # Depth of DiTBlocks
        num_heads=4,        # Attention Heads
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size

        # 1. 序列嵌入
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, input_len, hidden_size))

        # 2. 时间步嵌入
        self.t_embedder = TimestepEmbedder(hidden_size)

        # 3. Backbone
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        # 4. Final Layer
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, 1), # 输出 Voltage_reconstructed (1维)
        )
        self.adaLN_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

        self.initialize_weights()

    # xavier & Zero-Init
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.pos_embed, std=0.02)

        # Zero-Init
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.adaLN_final[-1].weight, 0)
        nn.init.constant_(self.adaLN_final[-1].bias, 0)

    def forward(self, x, t):
        """
        Inputs:
            x: (B, L, 2) 含噪电压 V_t， mask
            t: (B,) 扩散步数

        Outputs:
            voltage_reconstructed: (B, L, 1) 重建电压序列
        """
        # 1. Embedding
        x = self.x_embedder(x) + self.pos_embed

        # 2. Time Embedding
        c = self.t_embedder(t) # 全局条件向量现在只包含时间信息

        # 4. Transformer Forward
        for block in self.blocks:
            x = block(x, c)

        # 5. Output
        shift, scale = self.adaLN_final(c).chunk(2, dim=1)
        x = self.final_layer[0](x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.final_layer[1](x)
        velocity = self.final_layer[2](x)

        return velocity
