"""最小 LoRA-MoE 实现

与普通 MoE 的区别:
  - 普通 MoE: 每个专家是一个独立的完整 FFN, 参数量 = E * (full FFN)
  - LoRA-MoE: 所有专家共享一个冻结/共用的 base FFN, 每个专家只学一对低秩
              A, B 矩阵作为增量 (delta = B @ A, rank << dim)
  优点: 参数量只增加 E * 2 * rank * dim, 远小于 E 份完整 FFN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAExpert(nn.Module):
    """一个 LoRA 专家: 仅持有低秩增量 A, B; base 权重在外部共享传入"""

    def __init__(self, dim: int, hidden: int, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank

        # 对 base FFN 的两层分别加 LoRA 增量
        # W1: [dim -> hidden], LoRA: A1 [dim, r], B1 [r, hidden]
        self.A1 = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.B1 = nn.Parameter(torch.zeros(rank, hidden))   # 初始为 0, 保证初始增量为 0
        # W2: [hidden -> dim], LoRA: A2 [hidden, r], B2 [r, dim]
        self.A2 = nn.Parameter(torch.randn(hidden, rank) * 0.01)
        self.B2 = nn.Parameter(torch.zeros(rank, dim))

    def delta1(self, x: torch.Tensor) -> torch.Tensor:
        # x: [M, dim] -> [M, hidden]
        return (x @ self.A1) @ self.B1 * self.scaling

    def delta2(self, h: torch.Tensor) -> torch.Tensor:
        # h: [M, hidden] -> [M, dim]
        return (h @ self.A2) @ self.B2 * self.scaling


class LoRAMoE(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden: int,
        num_experts: int = 4,
        top_k: int = 2,
        rank: int = 4,
        alpha: float = 1.0,
        freeze_base: bool = True,
    ):
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k

        # 共享的 base FFN (所有 token 都会过)
        self.base_w1 = nn.Linear(dim, hidden)
        self.base_w2 = nn.Linear(hidden, dim)
        if freeze_base:
            for p in [self.base_w1.weight, self.base_w1.bias,
                      self.base_w2.weight, self.base_w2.bias]:
                p.requires_grad_(False)

        # 每个专家只是一对 LoRA 增量
        self.experts = nn.ModuleList(
            [LoRAExpert(dim, hidden, rank, alpha) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        x_flat = x.reshape(-1, d)            # [N, D]
        n = x_flat.size(0)

        # 1. 门控 + top-k
        logits = self.gate(x_flat)           # [N, E]
        topk_val, topk_idx = logits.topk(self.top_k, dim=-1)
        topk_w = F.softmax(topk_val, dim=-1)                   # [N, k]

        # 2. base 路径: 所有 token 都过共享 FFN 的第一层
        base_h = self.base_w1(x_flat)                           # [N, hidden]

        # 3. 累加每个被选中专家的 LoRA 增量 (第一层)
        h = base_h.clone()
        # 第二层在专家内独立计算 (因为激活在中间), 所以这里要按专家分组处理
        # 做法: 对每个专家 e, 取出路由到它的 token, 完整算 base+lora 的两层后, 加权写回
        out = torch.zeros_like(x_flat)
        for e in range(self.num_experts):
            mask = topk_idx == e
            if not mask.any():
                continue
            token_idx, slot_idx = mask.nonzero(as_tuple=True)
            xe = x_flat[token_idx]                              # [M, D]

            # 第一层: base + lora 增量, 然后激活
            h_e = self.base_w1(xe) + self.experts[e].delta1(xe)
            h_e = F.gelu(h_e)
            # 第二层: base + lora 增量
            y_e = self.base_w2(h_e) + self.experts[e].delta2(h_e)

            w = topk_w[token_idx, slot_idx].unsqueeze(-1)       # [M, 1]
            out.index_add_(0, token_idx, y_e * w)

        # 4. 负载均衡 aux loss
        probs = F.softmax(logits, dim=-1)
        P = probs.mean(dim=0)
        one_hot = F.one_hot(topk_idx, self.num_experts).float()
        f = one_hot.sum(dim=(0, 1)) / (n * self.top_k)
        aux_loss = self.num_experts * (f * P).sum()

        return out.reshape(b, t, d), aux_loss

    def trainable_param_count(self) -> tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D = 2, 5, 16
    moe = LoRAMoE(dim=D, hidden=32, num_experts=4, top_k=2, rank=4, alpha=8.0)

    x = torch.randn(B, T, D)
    y, aux = moe(x)
    print(f"input  : {x.shape}")
    print(f"output : {y.shape}")
    print(f"aux loss: {aux.item():.4f}")

    trainable, total = moe.trainable_param_count()
    print(f"params : trainable={trainable}, total={total}, ratio={trainable/total:.2%}")

    # 验证可反传 (只有 LoRA 参数和 gate 会拿到梯度)
    target = torch.randn_like(y)
    loss = F.mse_loss(y, target) + 0.01 * aux
    loss.backward()
    print(f"total loss: {loss.item():.4f}, grad ok")
