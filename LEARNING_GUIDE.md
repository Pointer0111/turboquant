# TurboQuant 学习路线

TurboQuant 是 ICLR 2026 论文 (arXiv:2504.19874) 的工程实现，核心是用近最优量化把 LLM 推理时的 KV Cache 压缩到约 3 bits/元素。

学习顺序：**先懂数学直觉，再按模块顺序读代码，最后跑验证实验**。

---

## 第一阶段：建立直觉（不看代码）

先理解"为什么这样设计"，否则代码会显得很魔法。

### 问题 1：为什么要旋转？

假设 key 向量 `x ∈ R^128`，各维度分布不均（某些维度方差很大）。直接量化会在小方差维度上浪费位数。随机正交旋转 `y = Πx` 把信息"均匀摊开"到每个维度，使每个坐标都服从同一个 Beta 分布，这样一套码本就能最优地量化所有坐标。这是论文 Theorem 1 的核心。

### 问题 2：为什么内积估计需要两阶段？

重建 `x̃` 的误差 `r = x - x̃` 虽然小，但会导致 `<q, x̃> ≠ <q, x>`，Attention score 有偏。QJL 用 `sign(Sr)` 的期望来补偿这个偏差，使估计值的期望等于真值（无偏估计，Theorem 2）。

估计式：
$$\langle q,\, x\rangle \approx \langle q,\, \tilde{x}_{\text{mse}}\rangle + \|r\|\cdot\frac{\sqrt{\pi/2}}{d}\cdot\langle S^{T}q,\,\text{sign}(Sr)\rangle$$

### 问题 3：为什么 Key 和 Value 用不同量化方案？

- **Key** 参与内积运算，决定 Attention 权重，需要无偏估计 → TurboQuantProd（两阶段）
- **Value** 只做加权求和，只需重建精度够用 → 简单 min-max 分组量化

### 问题 4：为什么要保留最近的 token 不压缩？

量化引入的误差对刚写入的 token 影响最大（这些 token 权重往往最高）。保留最近 128 个 token 的全精度副本（Ring Buffer），只压缩历史部分，可以在质量和压缩率之间取得平衡。

---

## 第二阶段：按顺序读代码

每个文件配一个动手实验，边读边跑。

### Step 1 — `turboquant/codebook.py`（约 30 分钟）

**读什么**：Beta 分布 PDF 的推导来源，以及 Lloyd-Max 迭代的两步：
- **E-step**：midpoint 更新决策边界（Voronoi 分区）
- **M-step**：`_conditional_mean` 把质心更新为该区间的条件期望

**动手实验**：

```python
from turboquant.codebook import compute_lloyd_max_codebook
import numpy as np

cb = compute_lloyd_max_codebook(d=128, bits=3)
print("质心位置:", np.round(cb['centroids'], 4))   # 8 个质心
print("决策边界:", np.round(cb['boundaries'], 4))  # 9 个边界（含 -1 和 +1）
print("每坐标 MSE:", cb['mse_per_coord'])

# 对比不同 bits 的压缩误差
for bits in [1, 2, 3, 4]:
    cb = compute_lloyd_max_codebook(d=128, bits=bits)
    print(f"bits={bits}: MSE/coord = {cb['mse_per_coord']:.4e}")
# 应验证 Theorem 3：误差按 1/4^b 缩减
```

**理解要点**：为什么 `d=128` 的码本不能直接用在 `d=64` 上（分布的形状参数 `(d-3)/2` 不同）。

---

### Step 2 — `turboquant/rotation.py`（约 15 分钟）

**读什么**：`generate_rotation_matrix` 和 `generate_qjl_matrix`。注意 QR 分解后修正符号的原因：让 `det(Q) = +1`，保证是旋转而非反射。

**动手实验**：

```python
import torch
from turboquant.rotation import generate_rotation_matrix, generate_qjl_matrix

Pi = generate_rotation_matrix(128, torch.device('cpu'))

# 验证正交性：Pi @ Pi^T 应为单位矩阵
print(torch.allclose(Pi @ Pi.T, torch.eye(128), atol=1e-5))   # True
print(torch.allclose(Pi.T @ Pi, torch.eye(128), atol=1e-5))   # True

# 验证旋转保持向量范数不变
x = torch.randn(10, 128)
y = x @ Pi.T
print(torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-4))  # True
```

**理解要点**：`rotate_forward(x, Pi)` 是 `x @ Pi^T`，`rotate_backward(y, Pi)` 是 `y @ Pi`，两者互为逆操作。

---

### Step 3 — `turboquant/quantizer.py`（核心，约 1-2 小时）

这是整个项目最重要的文件，建议分三步读：

**3a. 先读 bit-packing 工具函数**（`_pack_indices` / `_unpack_indices`）

理解 2-bit 的 4-pack、3-bit 升位到 4-bit 存储的原因（字节对齐），以及位移操作的方向。

```python
import torch
from turboquant.quantizer import _pack_indices, _unpack_indices

# 4 个 2-bit 索引打包进 1 个 uint8
indices = torch.tensor([[0, 1, 2, 3]])   # shape (1, 4)
packed  = _pack_indices(indices, bits=2)  # shape (1, 1)
print(packed)  # tensor([[228]], dtype=torch.uint8)  = 0b11100100

unpacked = _unpack_indices(packed, bits=2, d=4)
print(unpacked)  # 应还原为 [[0, 1, 2, 3]]
```

**3b. 读 `TurboQuantMSE`**

在纸上画出 `quantize` 的数据流：
```
x  →  norm()  →  x_unit  →  Pi @ x_unit  →  searchsorted  →  pack  →  indices
                   ↓
                 norms（存储以供还原）
```

**3c. 读 `TurboQuantProd`**（重点）

注意 `attention_score` 里的两项：MSE 项直接 matmul，QJL 项通过 `q @ S^T` 再与 sign bits 点积。

**动手实验**：

```python
import torch
from turboquant.quantizer import TurboQuantProd

q = TurboQuantProd(dim=128, bits=3, device=torch.device('cpu'))

# 模拟 10 个 key 向量
keys   = torch.randn(10, 128)
query  = torch.randn(1, 1, 128)

# 量化 keys
qkeys = q.quantize(keys.unsqueeze(0))  # (1, 10, 128) -> ProdQuantized

# 估计 Attention score
scores_approx = q.attention_score(query, qkeys).squeeze()      # (10,)
scores_exact  = (query.squeeze() @ keys.T)                      # (10,)

print("平均绝对误差:", (scores_approx - scores_exact).abs().mean().item())
print("相对误差:    ", ((scores_approx - scores_exact).abs() / scores_exact.abs().clamp(min=1e-6)).mean().item())

# 验证无偏性：多次随机，误差均值应接近 0
errors = []
for _ in range(100):
    keys  = torch.randn(100, 128)
    query = torch.randn(1, 1, 128)
    qkeys = q.quantize(keys.unsqueeze(0))
    err   = q.attention_score(query, qkeys).squeeze() - (query.squeeze() @ keys.T)
    errors.append(err.mean().item())
print("误差均值（应接近 0）:", sum(errors) / len(errors))
```

---

### Step 4 — `turboquant/kv_cache.py`（约 45 分钟）

**读什么**：

1. `quantize_values`：2-bit 的 4-pack 逻辑（注意与 `_pack_indices` 的区别：这里 4 个值按位域排列）
2. `TurboQuantKVCache.prefill` / `append` / `_flush_buffer`：理解"最近 buffer_size 个 token 精确保留，超出部分压缩"的滑动窗口机制

**动手实验**：

```python
import torch
from turboquant.kv_cache import quantize_values, dequantize_values

# 模拟一批 value 向量
v = torch.randn(4, 8, 128)   # (seq_len, heads, head_dim)

# 2-bit 量化
vq = quantize_values(v, bits=2, group_size=32)
print("原始大小 (bytes):", v.nelement() * 2)          # float16
print("压缩后大小 (bytes):", vq.data.nelement())        # uint8, 4x压缩

# 还原并测量误差
v_hat = dequantize_values(vq, group_size=32)
cos_sim = torch.nn.functional.cosine_similarity(v.reshape(-1, 128), v_hat.reshape(-1, 128)).mean()
print("cosine similarity:", cos_sim.item())   # 应接近 README 中的 0.940
```

---

### Step 5 — `turboquant/capture.py` + `turboquant/store.py`（约 45 分钟）

这两个文件配合着读，分别是**写路径**的两层：

- `RingBuffer`：固定容量的精确缓冲区，满了返回 overflow chunk（不丢弃，而是交给 store 压缩）
- `CompressedKVStore`：分块存储，写时只 append，不合并；读时懒惰 flatten（`_flat` 缓存）

**为什么分块存储？** 如果每次 decode 追加 1 个 token 都做 `torch.cat`，在 100k token 时会产生 O(n) 的内存分配开销。分块存储把合并推迟到第一次读操作。

**动手实验**：

```python
import torch
from turboquant.capture import RingBuffer

rb = RingBuffer(capacity=4, num_kv_heads=2, head_dim=64, device=torch.device('cpu'))

k = torch.randn(3, 2, 64)
v = torch.randn(3, 2, 64)

overflow = rb.write(k, v, num_tokens=3)
print("写入 3 个 token，overflow:", overflow)   # None，未溢出
print("buffer size:", rb.size)                   # 3

overflow = rb.write(k, v, num_tokens=3)
print("再写 3 个 token，overflow shape:", overflow[0].shape)   # (2, 2, 64)：溢出了 2 个
print("buffer size:", rb.size)                                  # 4（保留最新 4 个）
```

---

### Step 6 — `turboquant/score.py`（约 30 分钟）

**读什么**：`compute_hybrid_attention` 的分支逻辑（仅历史 / 仅最近 / 混合），以及 `_matmul_attend` 里的 GQA 处理。

**GQA 的关键实现**（`score.py:163-172`）：

```python
# 不用 repeat_interleave，用 broadcast 节省显存
q = query.view(T, H_kv, gqa_ratio, D).permute(1, 2, 0, 3)  # (H_kv, G, T, D)
k = kv_keys.unsqueeze(1)    # (H_kv, 1, N, D) — 在 G 维 broadcast
v = kv_values.unsqueeze(1)
scores = torch.einsum("hgtd,hgnd->hgtn", q, k) * scale
```

在 100k token 时，`repeat_interleave` 会把 KV 从 32 heads 扩展到 32*GQA heads，显存开销翻数倍；broadcast 不复制数据。

---

### Step 7 — `turboquant/triton_kernels.py`（可选，约 1 小时）

如果有 Triton 基础再读这部分。核心 insight：**不物化 dequant 向量**。

PyTorch 路径（慢）：
```
packed_indices → unpack → lookup centroids → 乘 Pi^T → 得到 key_dequant (d维) → 与 query 点积
```

Triton 路径（快）：
```
query_rot = q @ Pi^T  （一次性预计算）
直接对 packed_indices 逐字节解包，查码本，累加 query_rot[j] * centroid[idx[j]]
```

省去了生成完整 d 维向量的中间步骤，内存带宽减少 ~4x（读 3-bit 数据，不写 fp16 中间结果）。

第三个 fused kernel 实现了 Flash Attention 风格的 online softmax，一遍扫描完成 score + softmax + value 加权。

---

## 第三阶段：跑验证实验

代码读懂后，运行官方测试把输出和 README 的数字对上：

```bash
# 无需 GPU，验证论文 Theorem 1-3
python validate_paper.py

# 运行模块测试
python -m pytest test_modular.py -v

# 运行核心量化器测试
python -m pytest test_turboquant.py -v
```

**对照检查**：

| 测试 | 预期结果 | 对应代码位置 |
|------|----------|-------------|
| MSE 失真界 (Thm 1) | PASS，在 unit-norm 向量范围内 | `quantizer.py:TurboQuantMSE` |
| 无偏性 (Thm 2) | 相对偏差 < 0.1% | `quantizer.py:TurboQuantProd` |
| 失真 1/4^b 缩减 (Thm 3) | PASS | `codebook.py:compute_lloyd_max_codebook` |
| Recall@8 (3-bit, N=4096) | >= 0.40 | `quantizer.py:attention_score` |
| 压缩比 | ~4.41x (head_dim=256) | `kv_cache.py:memory_bytes` |

---

## 第四阶段：读 vLLM 集成（可选）

需要有 vLLM 使用经验再看这部分，不影响理解算法本身。

**`turboquant/integration/vllm.py`** 的核心是 `install_hooks`：

1. 遍历 `model_runner.compilation_config.static_forward_context`（vLLM 编译期静态图里所有 attention 层的字典）
2. 对每层的 `impl` 对象 monkey-patch 两个方法：
   - `do_kv_cache_update`：拦截 KV 写入，捕获到 TQ store
   - `forward`：在 decode 阶段可选地用 TQ 路径替换 flash attention

**四种运行模式**：

| 模式 | 行为 | 适用场景 |
|------|------|----------|
| `off` | 完全透传，无 TQ 活动 | 调试 |
| `capture_only` | 写入 TQ store，Attention 仍用 flash（默认） | 安全验证阶段 |
| `hybrid` | Decode 时用 TQ 压缩历史做 Attention | 生产使用 |
| `full_tq` | 预留，未实现 | 未来 |

`free_kv_cache()` 的工作原理：把 vLLM paged cache 的 tensor 替换成 1-byte dummy 张量，触发 CUDA 释放原来的显存块。

---

## 捷径：从测试文件反向学

如果不知道从哪里下手，直接读 `proof.py` 里的 A/B 对比测试，它把整个使用流程串联了起来。每看到一个调用，就去对应的实现里找细节。

---

## 核心数据结构速查

```
MSEQuantized
  ├── indices: uint8 tensor (bit-packed，每个坐标的码本索引)
  ├── norms:   原始向量的 L2 范数
  └── bits:    每个索引的位数

ProdQuantized
  ├── mse_indices:    uint8 tensor (MSE 部分，b-1 bits)
  ├── qjl_signs:      uint8 tensor (QJL 符号位，1 bit/维，8 个打包进 1 字节)
  ├── residual_norms: 残差向量的 L2 范数
  ├── norms:          原始向量的 L2 范数
  └── mse_bits:       MSE 部分的位数

ValueQuantized
  ├── data:   uint8 tensor (bit-packed，2-bit 时 4个值/字节)
  ├── scales: 每个量化组的 scale
  └── zeros:  每个量化组的 zero point

FlatCache
  ├── prod_q:     ProdQuantized (所有历史 key)
  ├── value_q:    ValueQuantized (所有历史 value)
  └── num_tokens: 历史 token 总数
```

---

## 推荐阅读顺序（极简版）

```
codebook.py → rotation.py → quantizer.py → kv_cache.py → capture.py → store.py → score.py
     ↓              ↓             ↓              ↓
  Lloyd-Max      正交旋转    两阶段量化器    值量化+缓冲
  Beta 分布      QJL 矩阵    (核心算法)      滑动窗口
```

每读完一个文件，跑一下对应的实验，确认和预期一致，再往下走。
