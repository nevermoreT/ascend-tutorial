# 实战案例：Qwen2.5-7B Attention 算子渐进式优化

> **硬件**：NVIDIA RTX 5080（Blackwell 架构，10752 CUDA Cores，336 × 5th Gen Tensor Cores，16GB GDDR7）
>
> **模型**：Qwen2.5-7B-Instruct
>
> **目标**：从零实现 Flash Attention 并集成到 vLLM，覆盖学习大纲第四~七篇核心知识点

---

## 1. 为什么选这个案例？

| 维度 | 理由 |
|------|------|
| **模型选择** | Qwen2.5-7B-Instruct，中文生态好，GQA（4 KV Heads）有优化空间，社区活跃 |
| **目标算子** | Attention — LLM 推理的绝对瓶颈（Prefill 阶段计算密集，Decode 阶段内存密集） |
| **硬件匹配** | 5080 Blackwell 5th Gen Tensor Core 原生支持 FP8/FP4，优化天花板高 |
| **与 CANN 的关系** | Attention 优化涉及内存层次/算子融合/调度策略，与 Runtime 开发直接相关 |

---

## 2. VRAM 预算评估

```
Qwen2.5-7B-Instruct FP16:
├── 模型权重:   ~15.2GB  ← 太紧，16GB 放不下 KV Cache
│
├── 方案 A: Qwen2.5-3B FP16 (~6GB) → 开发调试用，剩余 10GB 充裕
├── 方案 B: Qwen2.5-7B INT8  (~7.6GB) → bitsandbytes 量化，剩余 ~8GB
└── 方案 C: Qwen2.5-7B FP16 单层提取 → 算子级 benchmark，不需要加载完整模型
```

**推荐策略**：方案 C 做算子开发 → 方案 A 做集成验证 → 方案 B 做端到端 benchmark

---

## 3. 七阶段渐进路线

```
Stage 0        Stage 1         Stage 2          Stage 3          Stage 4          Stage 5          Stage 6
┌──────────┐  ┌──────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐
│ Profile  │→ │ Naive    │→  │ Tiled     │→  │ Flash     │→  │ Tensor    │→  │ Operator  │→  │ FP8 +     │
│ & Baseline│  │ CUDA     │   │ Attention │   │ Attention │   │ Core      │   │ Fusion    │   │ vLLM集成  │
└──────────┘  └──────────┘   └───────────┘   └───────────┘   └───────────┘   └───────────┘   └───────────┘
  认知问题      建立直觉      显存优化        终极形态        算力榨干        全链路优化      量产验证
```

---

## 4. 各阶段详解

### Stage 0：Profile & Baseline — 找到瓶颈

**目标**：在 Qwen2.5-3B 上跑通推理，用 Nsight 定位 Attention 的耗时占比

```python
# 核心工具链
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-3B-Instruct"
# 1. 加载模型，记录 baseline 吞吐
# 2. torch.profiler.profile() 录算子级耗时
# 3. nsys profile → Nsight Systems 可视化 timeline
```

**产出**：
- Prefill / Decode 两阶段的 token/s 基线
- 各算子耗时饼图（Attention 占多少？GEMM 占多少？）
- 确认优化目标：Prefill 看 Attention 的计算效率，Decode 看 KV Cache 的内存效率

**验收标准**：能回答 "Attention 在我的 5080 上到底慢在哪"

---

### Stage 1：Naive CUDA Attention — 建立直觉

**目标**：用最直白的 CUDA C++ 写一个 attention forward，理解为什么 naive 实现慢

```cpp
// Naive Attention Kernel
// 每个 thread 处理一个 (batch, head, q_seq) 的输出
__global__ void naive_attention(
    float* Q, float* K, float* V, float* O,
    int B, int H, int S_q, int S_kv, int D) {

    int b = blockIdx.x;
    int h = blockIdx.y;
    int q_idx = threadIdx.x + blockIdx.z * blockDim.x;

    // 1. Q·K^T → score[B,H,S_q,S_kv]
    // 2. softmax(score)
    // 3. score · V → O[B,H,S_q,D]
    // 问题：S=S_kv 时，中间 score 矩阵 O(N²) 显存，且无数据复用
}
```

**关键学习点**：
- Global Memory 的 `Q·K^T` 产生 `S_q × S_kv` 中间矩阵（序列长度 8192 时 = 512MB/层）
- 每次 softmax 都要读整个 S_kv 行，无数据复用
- 编译：`torch.utils.cpp_extension.load` 集成到 PyTorch

**验收标准**：功能正确（torch.allclose 验证），但比 `torch.nn.functional.scaled_dot_product_attention` 慢 5-10x

---

### Stage 2：Tiled Attention — 显存优化

**目标**：通过 Shared Memory Tiling 消除 O(N²) 中间矩阵

```cpp
// 核心思路：将 Q/K/V 切成 tile
// 每个 thread block 只加载一小块 Q_tile 和 K_tile 到 shared memory
// 分块计算 Q·K^T 的部分和，逐步累加

__shared__ float Q_tile[TILE_M][D];   // 一组 Q 向量
__shared__ float K_tile[TILE_N][D];   // 一组 K 向量

// 分块累加 softmax 的 numerator 和 denominator
// → 不再需要完整 S_q × S_kv 矩阵
```

**关键学习点**：
- Shared Memory 是程序员管理的 L1 Cache（5080 每个 SM 128KB）
- Bank Conflict 的识别与规避
- 分块累加 partial softmax 的数值稳定性

**测量**：
- 显存占用：从 O(N²) → O(TILE_M × TILE_N)，长序列（8192+）可跑
- 速度：可能仍比 PyTorch SDPA 慢 2-3x，但理解了关键瓶颈

**验收标准**：序列长度 8192 时，naive 版 OOM 而 tiled 版正常运行

---

### Stage 3：Flash Attention — 终极形态

**目标**：实现 Online Softmax + Block Tiling = Flash Attention 的核心思想

```cpp
// Flash Attention 的两个关键创新：
// 1. Online Softmax: 不需要先读完所有 score 再做 softmax
//    维护 running max(m_j) 和 running sum(l_j)
//    每处理一个新的 K/V block，修正之前的结果
//
// 2. Block Tiling across K/V sequence dimension
//    外层循环遍历 K/V blocks（不需要完整矩阵）
//    内层循环遍历 Q blocks

for (int j = 0; j < num_kv_blocks; j++) {
    load_K_tile(K, j);
    load_V_tile(V, j);                   // HBM → SRAM

    S_tile = Q_tile @ K_tile.T;          // SRAM 内计算
    m_new = max(m_old, max(S_tile));      // running max
    l_new = l_old * exp(m_old - m_new)
          + sum(exp(S_tile - m_new));
    O_tile = O_tile * (l_old / l_new) * exp(m_old - m_new)
           + (exp(S_tile - m_new) / l_new) @ V_tile;

    m_old = m_new;
    l_old = l_new;
}
```

**关键学习点**：
- Online Softmax 数值推导（为什么要维护 m 和 l）
- SRAM 计算量与 HBM 访问量的 trade-off
- 反向传播的推导（checkpointing，不存储中间 attention matrix）

**参考论文**：*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (Dao et al., 2022)

**验收标准**：
- 功能等价于 `F.scaled_dot_product_attention`（误差 < 1e-3）
- 序列长度 16384 可跑
- 显存占用线性于序列长度（而非平方）

---

### Stage 4：Tensor Core 加速 — 算力榨干

**目标**：用 WMMA API 或 `nvcuda::wmma` 把 `Q·K^T` 和 `score·V` 扔给 Tensor Core

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

// Q_tile[16×128] @ K_tile^T[128×16] → S_tile[16×16]
// 利用 5th Gen Tensor Core，FP16 矩阵乘吞吐翻倍

fragment<matrix_a, 16, 16, 16, half, row_major> Q_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> K_frag;
fragment<accumulator, 16, 16, 16, half> S_frag;

load_matrix_sync(Q_frag, Q_shared, D);    // SRAM → Tensor Core register
load_matrix_sync(K_frag, K_shared, D);
fill_fragment(S_frag, 0.0f);
mma_sync(S_frag, Q_frag, K_frag, S_frag); // Tensor Core 一拍完成 16×16×16
```

**关键学习点**：
- Tensor Core 的矩阵形状约束（16×16×16 / 16×8×32 等）
- 数据从 Shared Memory → Tensor Core Register 的搬运
- Blackwell 5th Gen Tensor Core 的 FP8 能力（下一阶段的伏笔）

**验收标准**：Attention kernel TFLOPS 利用率 > 40%（Nsight Compute 测量）

---

### Stage 5：Operator Fusion — 全链路优化

**目标**：将 Attention 前后的操作融合为单一 kernel，减少 kernel launch 开销和显存往返

```
// 原始流程（6次 kernel launch，5次显存写回）:
Q = input @ Wq          // GEMM → HBM
K = input @ Wk          // GEMM → HBM
V = input @ Wv          // GEMM → HBM
Q = apply_rope(Q)       // Elementwise → HBM
O = flash_attn(Q,K,V)   // Flash Attn → HBM
O = O @ Wo              // GEMM → HBM

// 融合后（2次 kernel launch，1次显存写回）:
// Kernel 1: QKV Projection + RoPE Fusion
//   Q = RoPE(input @ Wq)  ─┐
//   K = RoPE(input @ Wk)  ─┤ 全部在 SRAM，不写回 HBM
//   V = input @ Wv        ─┘
//   → 直接送入 Flash Attention
// Kernel 2: Flash Attention + O Projection
//   O = flash_attn(Q,K,V)
//   O = O @ Wo             // epilogue fusion
//   → 最终写回 HBM
```

**关键学习点**：
- CUTLASS Epilogue Fusion 模式
- `cp.async` 异步数据搬运 + 双缓冲
- Register pressure 与 occupancy 的权衡

**验收标准**：端到端 Attention 层延迟比 Stage 0 baseline 快 2x+

---

### Stage 6：FP8 量化 + vLLM 集成 — 量产验证

**目标**：利用 Blackwell FP8 Tensor Core，将优化后的算子集成到 vLLM

```python
# 1. FP8 量化 Qwen2.5-7B
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float8_e4m3fn  # Blackwell 原生支持
)

# 2. 注册自定义 Attention 到 vLLM
# vLLM 的 Attention 后端接口:
# - validate_request()
# - forward() → 替换为你的 Flash Attention kernel

# 3. benchmark
# vLLM 自带 benchmark_serving.py
# 对比: 原生 vLLM vs 你的自定义 kernel
```

**关键学习点**：
- FP8（E4M3/E5M2）的数值范围与精度损失评估
- vLLM 的 PagedAttention 接口与自定义算子注册
- 端到端吞吐量 benchmark 方法论

**验收标准**：
- vLLM 端到端 throughput（tokens/s）提升可量化
- 长序列（8K+）场景提升显著
- 数值精度在可接受范围（困惑度下降 < 0.5%）

---

## 5. 各阶段预期收益总览

| Stage | 优化手段 | 预期提升 | 核心学习价值 |
|-------|---------|---------|-------------|
| 0 | Profile | 建立基线 | Nsight 工具链 |
| 1 | Naive CUDA | 慢 5-10x | CUDA 编程入门 |
| 2 | Tiled | 显存 ↓↓ | 内存层次、Bank Conflict |
| 3 | Flash Attn | 显存 ↓↓↓，速度 ↑ 2-3x | Online Softmax、IO-Awareness |
| 4 | Tensor Core | 速度 ↑ 2-4x | WMMA/PTX MMA、混合精度 |
| 5 | Op Fusion | 延迟 ↓ 30-50% | Kernel 融合、异步流水线 |
| 6 | FP8 + vLLM | 吞吐 ↑，显存 ↓ 50% | 量化、框架集成 |

---

## 6. 开发环境

```bash
# Docker 一键环境（基于 PyTorch 官方镜像）
docker run --gpus all -it --rm \
  pytorch/pytorch:2.5.1-cuda12.8-cudnn9-devel

# 关键依赖
pip install transformers vllm sglang

# CUDA Toolkit 12.8+ (支持 Blackwell)
# Nsight Systems / Nsight Compute (CUDA Toolkit 自带)
```

**验证环境就绪**：

```bash
nvidia-smi          # 确认 5080 被识别
nvcc --version      # CUDA 12.8+
nsys --version      # Nsight Systems
ncu --version       # Nsight Compute
python -c "import torch; print(torch.cuda.get_device_name())"
# 期望输出: NVIDIA GeForce RTX 5080
```
