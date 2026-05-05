# Triton Kernel 接入生产路径 — 实施计划

## 目标

将 `triton_kernels.py` 里的 `turboquant_fused_decode` 接入 `score.py` 的推理路径，替换掉当前"先 dequantize 成 fp32，再做 matmul"的 PyTorch fallback。

完成后，decode 阶段不再物化完整 fp16/fp32 KV 向量，直接在 packed bits 上完成 score + softmax + value 加权，内存带宽降低约 4x。

---

## 当前状态（改动前）

**`score.py` 两处问题代码：**

```python
# _attend_compressed_only (line 93)
k_dequant = quantizer.dequantize(flat.prod_q)   # ← 展开成完整 fp32，浪费
v_dequant = dequantize_values(flat.value_q, 32)
return _matmul_attend(query, k_dequant, v_dequant, ...)

# _attend_hybrid (line 126-135)
k_hist = quantizer.dequantize(flat.prod_q)       # ← 同上
v_hist = dequantize_values(flat.value_q, 32)
k_all = torch.cat([k_hist.float(), k_recent.float()], dim=1)  # ← concat 再算
v_all = torch.cat([v_hist.float(), v_recent.float()], dim=1)
return _matmul_attend(query, k_all, v_all, ...)
```

**`triton_kernels.py` 的两个缺陷（不改动无法接入）：**

1. `_turboquant_fused_decode_kernel` 在最后一行做了归一化（`acc = acc / l_i`），丢掉了 softmax 状态 `(m_i, l_i)`，导致无法与 recent buffer 做 log-sum-exp 合并。
2. kernel 的 grid 维度 `pid_bh` 对应 KV head，不支持 GQA（query heads > kv heads 的情况）。

---

## 三个需要修改的文件

| 文件 | 改动内容 |
|------|----------|
| `turboquant/triton_kernels.py` | 1. kernel 增加 `RETURN_STATE` 模式，返回 softmax 状态 <br> 2. kernel 增加 `GQA_RATIO` 参数，支持 query head ≠ kv head |
| `turboquant/score.py` | 替换 `_attend_compressed_only` 和 `_attend_hybrid`，加 Triton 可用性检测和 fallback |
| `turboquant/__init__.py` | 无需改动 |

---

## Phase 1 — 修改 kernel：支持返回 softmax 状态

**文件**：`turboquant/triton_kernels.py`

**改动位置**：`_turboquant_fused_decode_kernel`，在参数列表末尾加一个 constexpr 开关：

```python
RETURN_STATE: tl.constexpr = False,   # 新增参数
```

在 kernel 结尾，把原来无条件的归一化改成：

```python
# 原来（固定归一化）：
acc = acc / l_i
tl.store(OUT_ptr + ..., acc)

# 改成：
if RETURN_STATE:
    # 不归一化，把 acc / m_i / l_i 分别存出
    tl.store(OUT_ptr  + pid_bh * stride_o_bh + d_offs * stride_o_d, acc)
    tl.store(M_ptr    + pid_bh, m_i)
    tl.store(L_ptr    + pid_bh, l_i)
else:
    acc = acc / l_i
    tl.store(OUT_ptr  + pid_bh * stride_o_bh + d_offs * stride_o_d, acc)
```

同时在参数列表里加 `M_ptr` 和 `L_ptr`（`RETURN_STATE=False` 时传哑指针，kernel 内部不 store）。

然后在 Python wrapper 里新增一个函数：

```python
def turboquant_fused_decode_with_state(
    query, quantized_key, value_quantized,
    Pi, S, centroids, mse_bits, qjl_scale, sm_scale, group_size=32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    和 turboquant_fused_decode 相同，但返回 (acc_unnorm, m, l)。
    acc_unnorm: (BH, D) — 未归一化的加权 value 累加
    m:          (BH,)   — 每个 head 的 running max
    l:          (BH,)   — 每个 head 的 running sum of exp
    """
    # ... 和现有 wrapper 基本相同，但调用 kernel 时 RETURN_STATE=True
    # ... 多分配两个输出 tensor: m (BH,), l (BH,)
```

原来的 `turboquant_fused_decode` 保持不变（`RETURN_STATE=False`），保证已有代码不破坏。

---

## Phase 2 — 修改 kernel：支持 GQA

**文件**：`turboquant/triton_kernels.py`

**改动位置**：`_turboquant_fused_decode_kernel`，增加：

```python
GQA_RATIO: tl.constexpr = 1,   # 新增参数，默认 1（无 GQA）
```

把 kernel 的 grid 从 `(BH_kv,)` 改成 `(BH_q,)`，grid 总大小 = `BH_kv * GQA_RATIO`。
在 kernel 内部用：

```python
pid_bh_q  = tl.program_id(0)          # query head 索引
pid_bh_kv = pid_bh_q // GQA_RATIO     # 对应的 kv head 索引
```

所有读 KV 数据的指针偏移改用 `pid_bh_kv`，读 query 的指针偏移用 `pid_bh_q`，
输出写到 `pid_bh_q`。

Python wrapper 里改 grid 调用：

```python
grid = (BH_q,)   # 原来是 (BH_kv,)，现在按 query heads 并行
```

同样地，`turboquant_fused_decode_with_state` 也支持 `gqa_ratio` 参数。

---

## Phase 3 — 修改 score.py：接入 Triton

**文件**：`turboquant/score.py`

### 3a. 加 Triton 可用性检测

在文件顶部加：

```python
try:
    import triton
    from turboquant.triton_kernels import (
        turboquant_fused_decode,
        turboquant_fused_decode_with_state,
    )
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False
```

### 3b. 新增辅助函数：recent buffer 的 softmax 状态

```python
def _exact_attention_with_state(
    query: torch.Tensor,       # (BH_q, D)
    recent_k: torch.Tensor,   # (BH_kv, N_recent, D)
    recent_v: torch.Tensor,   # (BH_kv, N_recent, D)
    gqa_ratio: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    返回 exact recent buffer 的 softmax 状态，用于与 Triton 结果合并。
    返回: (acc_unnorm, m, l)，形状均为 (BH_q, D) / (BH_q,) / (BH_q,)
    """
    BH_kv, N_recent, D = recent_k.shape
    # GQA: 把 kv 扩展到 query heads
    k = recent_k.repeat_interleave(gqa_ratio, dim=0)   # (BH_q, N_recent, D)
    v = recent_v.repeat_interleave(gqa_ratio, dim=0)

    scores = torch.bmm(query.unsqueeze(1), k.transpose(1, 2)).squeeze(1) * scale
    # scores: (BH_q, N_recent)

    m = scores.max(dim=-1).values                        # (BH_q,)
    p = torch.exp(scores - m.unsqueeze(-1))              # (BH_q, N_recent)
    l = p.sum(dim=-1)                                    # (BH_q,)
    acc = torch.bmm(p.unsqueeze(1), v).squeeze(1)        # (BH_q, D)，未归一化

    return acc, m, l
```

### 3c. 新增 log-sum-exp 合并函数

```python
def _merge_softmax_states(
    acc_a: torch.Tensor, m_a: torch.Tensor, l_a: torch.Tensor,
    acc_b: torch.Tensor, m_b: torch.Tensor, l_b: torch.Tensor,
) -> torch.Tensor:
    """
    合并两段注意力的 softmax 状态。
    数学上等价于把两段的 logits 拼接后做一次 softmax。
    
    m_global = max(m_a, m_b)
    l_global = l_a * exp(m_a - m_global) + l_b * exp(m_b - m_global)
    out      = (acc_a * l_a * exp(m_a - m_global)
              + acc_b * l_b * exp(m_b - m_global)) / l_global
    """
    m_global = torch.maximum(m_a, m_b)                       # (BH,)
    scale_a  = torch.exp(m_a - m_global).unsqueeze(-1)       # (BH, 1)
    scale_b  = torch.exp(m_b - m_global).unsqueeze(-1)
    l_a_s    = (l_a * scale_a.squeeze(-1))                   # (BH,)
    l_b_s    = (l_b * scale_b.squeeze(-1))
    l_global = l_a_s + l_b_s                                  # (BH,)

    out = (acc_a * scale_a + acc_b * scale_b) / l_global.unsqueeze(-1)
    return out                                                 # (BH, D)
```

### 3d. 替换 `_attend_compressed_only`

```python
def _attend_compressed_only(query, flat, quantizer, gqa_ratio, num_kv_heads, scale):
    if _TRITON_AVAILABLE and query.is_cuda:
        # query: (T, Q, D) — decode 时 T=1
        T, Q, D = query.shape
        BH_q = T * Q
        q_flat = query.reshape(BH_q, D)

        out = turboquant_fused_decode(
            query          = q_flat.unsqueeze(1),
            quantized_key  = flat.prod_q,
            value_quantized = flat.value_q,
            Pi             = quantizer.mse_quantizer.Pi,
            S              = quantizer.S,
            centroids      = quantizer.mse_quantizer.centroids,
            mse_bits       = quantizer.bits - 1,
            qjl_scale      = quantizer.qjl_scale,
            sm_scale       = scale,
            group_size     = 32,
            gqa_ratio      = gqa_ratio,
        )  # (BH_q, D)

        return out.reshape(T, Q, D).to(query.dtype)

    # fallback
    k_dequant = quantizer.dequantize(flat.prod_q)
    v_dequant = dequantize_values(flat.value_q, 32)
    return _matmul_attend(query, k_dequant, v_dequant, gqa_ratio, num_kv_heads, scale)
```

### 3e. 替换 `_attend_hybrid`

```python
def _attend_hybrid(query, flat, quantizer, recent_k, recent_v,
                   gqa_ratio, num_kv_heads, head_dim, scale):
    if _TRITON_AVAILABLE and query.is_cuda:
        T, Q, D = query.shape
        BH_q = T * Q
        BH_kv = num_kv_heads
        q_flat = query.reshape(BH_q, D)

        # 历史段：Triton fused，返回 softmax 状态
        acc_hist, m_hist, l_hist = turboquant_fused_decode_with_state(
            query          = q_flat.unsqueeze(1),
            quantized_key  = flat.prod_q,
            value_quantized = flat.value_q,
            Pi             = quantizer.mse_quantizer.Pi,
            S              = quantizer.S,
            centroids      = quantizer.mse_quantizer.centroids,
            mse_bits       = quantizer.bits - 1,
            qjl_scale      = quantizer.qjl_scale,
            sm_scale       = scale,
            group_size     = 32,
            gqa_ratio      = gqa_ratio,
        )  # (BH_q, D), (BH_q,), (BH_q,)

        # recent 段：PyTorch exact，返回 softmax 状态
        k_kv = recent_k.transpose(0, 1)  # (BH_kv, N_recent, D)
        v_kv = recent_v.transpose(0, 1)
        acc_rec, m_rec, l_rec = _exact_attention_with_state(
            q_flat, k_kv, v_kv, gqa_ratio, scale
        )

        # log-sum-exp 合并
        out = _merge_softmax_states(acc_hist, m_hist, l_hist,
                                    acc_rec,  m_rec,  l_rec)
        return out.reshape(T, Q, D).to(query.dtype)

    # fallback（原有逻辑不变）
    k_hist = quantizer.dequantize(flat.prod_q)
    v_hist = dequantize_values(flat.value_q, 32)
    k_all = torch.cat([k_hist.float(), recent_k.transpose(0,1).float()], dim=1)
    v_all = torch.cat([v_hist.float(), recent_v.transpose(0,1).float()], dim=1)
    return _matmul_attend(query, k_all, v_all, gqa_ratio, num_kv_heads, scale)
```

---

## Phase 4 — 测试

### 4a. 数值正确性测试

新建 `test_triton_integration.py`，验证三件事：

**测试 1：`_attend_compressed_only` Triton 路径与 fallback 路径输出一致**

```python
# 在 CUDA 上分别用 Triton 和 PyTorch 算同一批数据，误差应 < 1e-3
out_triton = _attend_compressed_only(query, flat, quantizer, gqa_ratio=1, ...)
out_pytorch = ...  # 强制走 fallback
assert (out_triton - out_pytorch).abs().max() < 1e-3
```

**测试 2：`_attend_hybrid` 的 log-sum-exp 合并数学正确**

```python
# 合并路径应与"concat 再 softmax"的结果一致（差异仅来自量化误差）
out_merged = _attend_hybrid(query, flat, quantizer, recent_k, recent_v, ...)
out_concat  = ...  # fallback 的 concat 路径
assert (out_merged - out_concat).abs().max() < 1e-2
```

**测试 3：GQA 正确（gqa_ratio > 1）**

```python
# 用 gqa_ratio=4，num_kv_heads=8，num_query_heads=32
# Triton 路径与 fallback 路径结果一致
```

### 4b. 边界情况

| 情况 | 预期行为 |
|------|----------|
| Triton 未安装 | 自动走 fallback，不报错 |
| `num_tokens == 1`（单 token decode） | 正常运行 |
| `N < BLOCK_N`（历史 token 数少于块大小） | kernel 内 mask 处理，不越界 |
| `gqa_ratio == 1`（无 GQA） | 等价于原有行为 |
| recent buffer 为空（`has_recent=False`） | 走 `_attend_compressed_only`，不触发 hybrid |

### 4c. 性能对比

在完成正确性验证后，跑 `proof.py` 对比改动前后的吞吐，预期：
- 显存峰值下降（不再物化 fp32 KV）
- decode tok/s 提升（更少内存带宽）

---

## 实施顺序

```
Phase 1（kernel 加 RETURN_STATE）
    ↓
Phase 3d（_attend_compressed_only 接 Triton）
    ↓ 先跑测试 4a 测试1，确认压缩历史单段正确
Phase 2（kernel 加 GQA_RATIO）
    ↓
Phase 3e（_attend_hybrid 接 Triton + log-sum-exp）
    ↓ 跑测试 4a 测试2和3
Phase 4b/4c（边界 + 性能）
```

建议每个 Phase 对应一个独立 commit，方便二分定位问题。

---

## 风险点

**风险 1：Triton constexpr 限制**

`D`、`PACKED_D_MSE` 等是 `tl.constexpr`，意味着每种不同的 `(D, bits, group_size)` 组合都会触发一次 JIT 编译。新增的 `GQA_RATIO` 也是 `constexpr`，若有多种 gqa_ratio 值会多编译几次。首次推理延迟会增加，后续走缓存无影响。

**风险 2：`RETURN_STATE` 模式下哑指针的处理**

`RETURN_STATE=False` 时，`M_ptr` 和 `L_ptr` 需要传合法的 CUDA 指针（哑 tensor，不写入），否则 Triton 在地址合法性检查时报错。Python wrapper 里分配一个 size=1 的临时 tensor 即可。

**风险 3：recent buffer 的 GQA 扩展**

`_exact_attention_with_state` 里用了 `repeat_interleave(gqa_ratio)` 复制 KV，会临时多占 `gqa_ratio` 倍的 recent buffer 显存。recent buffer 默认 128 token，即使 gqa_ratio=8 也只是 128×8×256×2 ≈ 0.5 MB，可忽略。若 ring buffer 非常大则需评估。
