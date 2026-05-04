"""最小 MoE (Mixture of Experts) 实现

核心思想:
  - 多个 "专家" (Expert) 是结构相同的小型 FFN
  - 一个 "门控网络" (Gate) 为每个 token 选出 top-k 个专家
  - 只有被选中的专家参与计算, 输出按门控权重加权求和
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """一个专家就是一个普通的两层 FFN"""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoE(nn.Module):
    def __init__(self, dim: int, hidden: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([Expert(dim, hidden) for _ in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, D] -> 拍平成 [N, D] 方便按 token 路由
        b, t, d = x.shape
        x_flat = x.reshape(-1, d)            # [N, D], N = B*T

        # 1. 门控打分 + top-k 选择
        logits = self.gate(x_flat)           # [N, E]
        topk_val, topk_idx = logits.topk(self.top_k, dim=-1)   # [N, k]
        topk_w = F.softmax(topk_val, dim=-1)                   # [N, k] 仅对被选中的专家归一化

        # 2. 把每个 token 路由到选中的专家, 收集输出
        out = torch.zeros_like(x_flat)
        for e in range(self.num_experts):
            # mask: 哪些 (token, slot) 落到了专家 e
            mask = topk_idx == e             # [N, k]
            if not mask.any():
                continue
            token_idx, slot_idx = mask.nonzero(as_tuple=True)  # 落到该专家的 token 位置
            expert_in = x_flat[token_idx]                      # [M, D]
            expert_out = self.experts[e](expert_in)            # [M, D]
            w = topk_w[token_idx, slot_idx].unsqueeze(-1)      # [M, 1]
            out.index_add_(0, token_idx, expert_out * w)

        # 3. 负载均衡辅助损失 (Switch Transformer 风格)
        #    - f_i: 专家 i 实际接收 token 的比例
        #    - P_i: 专家 i 的平均门控概率
        #    - aux = E * sum(f_i * P_i), 鼓励两者都接近 1/E
        probs = F.softmax(logits, dim=-1)                      # [N, E]
        P = probs.mean(dim=0)                                  # [E]
        one_hot = F.one_hot(topk_idx, self.num_experts).float()  # [N, k, E]
        f = one_hot.sum(dim=(0, 1)) / (x_flat.size(0) * self.top_k)  # [E]
        aux_loss = self.num_experts * (f * P).sum()

        return out.reshape(b, t, d), aux_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D = 2, 5, 16
    moe = MoE(dim=D, hidden=32, num_experts=4, top_k=2)

    x = torch.randn(B, T, D)
    y, aux = moe(x)
    print(f"input  : {x.shape}")
    print(f"output : {y.shape}")
    print(f"aux loss: {aux.item():.4f}")

    # 简单训练一步, 验证可反传
    target = torch.randn_like(y)
    loss = F.mse_loss(y, target) + 0.01 * aux
    loss.backward()
    print(f"total loss: {loss.item():.4f}, grad ok")
