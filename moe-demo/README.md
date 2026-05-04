# MoE Demo

最小可运行的 Mixture of Experts 实现。

## 文件

- [moe.py](moe.py) — 标准 MoE：每个专家是独立的两层 FFN，门控 top-k 路由 + 负载均衡 aux loss
- [lora_moe.py](lora_moe.py) — LoRA-MoE：所有专家共享一个冻结的 base FFN，每个专家只学一对低秩 A/B 矩阵作为增量

## 运行

```bash
pip install torch
python moe.py
python lora_moe.py
```

两个脚本的 `__main__` 都会跑一次前向 + 反向，打印输出形状、aux loss 和总 loss，验证梯度正常。

## 关键点

- **门控**：`gate(x)` → top-k 选专家 → softmax 归一化作为加权
- **稀疏激活**：每个 token 只过 `k` 个专家，未被选中的专家不参与该 token 的计算
- **负载均衡**：`aux_loss = E * Σ(f_i * P_i)`，鼓励 token 在专家间均匀分布
- **LoRA 版优势**：参数量从 `E × full_FFN` 降到 `E × 2 × rank × dim`
