# GPU/NPU 软件编程 Tutorial — 学习大纲

> **定位**：面向两类人群的渐进式教程
>
> - **路径 A（入门者）**：从零建立异构计算认知，掌握 CUDA/NPU 编程基础
> - **路径 B（进阶者）**：深入算子优化与推理加速，达到工业级实践能力
>
> **技术栈**：PyTorch 全栈 · vLLM + SGLang · 华为昇腾 CANN

---

## 第一篇：异构计算硬件演进史（路径 A/B 共修）

| 章节 | 主题 | 内容 |
|------|------|------|
| 01 | GPU 发展历程 | 从图形加速器到 AI 算力核心（G80 → Fermi → Volta → Ampere → Hopper → Blackwell） |
| 02 | TPU 发展历程 | Google TPU v1~v5 架构演进，脉动阵列设计哲学 |
| 03 | NPU 发展历程 | 华为昇腾 Da Vinci 架构、寒武纪、苹果 Neural Engine、高通 Hexagon |
| 04 | AI 芯片格局总览 | 算力/带宽/功耗横评，训练芯片 vs 推理芯片设计权衡 |
| 05 | 并行计算模型基础 | SIMT/SIMD/SPMD 辨析，Amdahl 定律与 Gustafson 定律 |

---

## 第二篇：LLM 计算基础 — 我们在优化什么（路径 A/B 共修）

> **设计意图**：在学习 CUDA 编程之前，先理解 LLM 的计算结构。只有知道"模型长什么样"，才能理解"为什么要这样优化硬件"。

| 章节 | 主题 | 内容 |
|------|------|------|
| 06 | Transformer 架构全景 | Encoder-Decoder / Decoder-Only 架构，Tokenization → Embedding → Transformer Block → LM Head 全流程 |
| 07 | Attention 机制详解 | Scaled Dot-Product Attention 数学推导，Q/K/V 的物理含义，Multi-Head Attention (MHA) 分头计算与拼接 |
| 08 | Attention 变体与演进 | **GQA**（Grouped-Query Attention）、**MQA**（Multi-Query Attention）、MLA（Multi-head Latent Attention），KV Cache 压缩思路 |
| 09 | FFN 与激活函数 | Position-wise FFN 结构，GELU → SiLU → SwiGLU 演进，MoE（Mixture of Experts）的 FFN 替代 |
| 10 | 位置编码 | 绝对位置编码（Sinusoidal / Learned）→ **RoPE**（旋转位置编码）→ ALiBi，RoPE 的旋转矩阵推导与代码实现 |
| 11 | LLM 推理的两个阶段 | **Prefill**（并行处理 prompt，计算密集）vs **Decode**（逐 token 自回归，内存密集），KV Cache 的生命周期 |
| 12 | 从模型到硬件：计算图分析 | Transformer 各层的 GEMM 分解（QKV 投影、Attention Score、O 投影、FFN），Prefill/Decode 各阶段的算力与带宽需求量化分析 |
| 13 | 经典模型架构拆解 | GPT-2 / LLaMA / Qwen / DeepSeek-V3 架构对比，参数量估算，显存占用计算公式 |

---

## 第三篇：CUDA 编程基础（路径 A 核心）

| 章节 | 主题 | 内容 |
|------|------|------|
| 14 | CUDA 编程模型 | Grid/Block/Thread 层次，Warp 调度机制，线程束分化 |
| 15 | CUDA 内存体系 | Global/Shared/Constant/Texture/Register，缓存层次，合并访问模式 |
| 16 | CUDA 同步与原子操作 | `__syncthreads()`、原子函数、Warp 级原语（`__shfl_sync`） |
| 17 | CUDA Stream 与 Event | 多流并发、Event 计时、流水线隐藏延迟 |
| 18 | CUDA 错误处理与调试 | `cudaGetLastError`、`cuda-memcheck`、compute sanitizer |

---

## 第四篇：CUDA 进阶优化（路径 B 核心）

| 章节 | 主题 | 内容 |
|------|------|------|
| 19 | 性能 Profiling | Nsight Systems / Nsight Compute 实战，Roofline Model 瓶颈定位 |
| 20 | Tensor Core 编程 | WMMA API、MMA PTX 级操作、混合精度（FP16/BF16/FP8/INT8） |
| 21 | CUDA 协作组与异步 | Cooperative Groups、CUDA Graph、Async Copy（`cp.async`） |
| 22 | 多 GPU 编程 | NCCL 通信原语、NVLink/PCIe 拓扑感知、张量并行通信模式 |

---

## 第五篇：算子开发实战 — 以 LLM 算子为核心（路径 A/B 分流 → 汇合）

| 章节 | 主题 | 内容 | 路径 |
|------|------|------|------|
| 23 | 算子设计方法论 | 计算密集 vs 内存密集，Roofline 分析性能上限 | A+B |
| 24 | Elementwise 算子 | 向量运算、激活函数（GELU/SiLU/SwiGLU），kernel fuse | A+B |
| 25 | Reduction 算子 | Warp Reduce → Block Reduce → Grid Reduce，避免 bank conflict | A+B |
| 26 | GEMM 算子（上） | 分块 Tiling 策略、Shared Memory 双缓冲 | B |
| 27 | GEMM 算子（下） | 寄存器分块、Epilogue fusion、Tensor Core 加速 | B |
| 28 | QKV 投影算子 | Batched GEMM（Q/K/V 三路并行）、RoPE 融合、bias fusion | B |
| 29 | Flash Attention（上） | Prefill 场景：分块策略、Online Softmax、前向传播实现 | B |
| 30 | Flash Attention（下） | Decode 场景：KV Cache 读取模式、GQA/MQA 优化、反向传播 | B |
| 31 | FFN 算子 | Gate/Up/Down 三路 GEMM 融合、SiLU 激活融合、MoE 路由算子 | B |
| 32 | KV Cache 管理 | Paged KV Cache 内存布局、Block Table 管理逻辑、Prefix Caching | B |
| 33 | 卷积算子（选修） | im2col + GEMM、Winograd、Implicit GEMM 直接卷积 | B |

---

## 第六篇：算子开发框架与 LLM 硬件协同（路径 B）

| 章节 | 主题 | 内容 |
|------|------|------|
| 34 | Triton 编程 | Block 编程范式、与 CUDA 对比，用 Triton 写 Flash Attention |
| 35 | CUTLASS 库 | 模板化 GEMM 架构、Epilogue fusion（GELU/SiLU/Bias 融合） |
| 36 | TransformerEngine | FP8 量化引擎、延迟缩放（Delay Scaling）、与 PyTorch 集成 |
| 37 | cuDNN / cuBLAS / FlashInfer | 高性能库使用、FlashInfer 的 Attention kernel 调用 |
| 38 | 算子自动生成简介 | TVM / Tensor Expression / Ansor 自动调优 |

---

## 第七篇：PyTorch 异构计算集成

| 章节 | 主题 | 内容 |
|------|------|------|
| 39 | PyTorch CUDA 扩展 | `torch.utils.cpp_extension`，自定义 CUDA op 注册 |
| 40 | PyTorch `torch.compile` | TorchInductor 后端、自定义后端接入 |
| 41 | PyTorch 分布式训练 | DDP / FSDP 原理，通信与计算重叠 |
| 42 | PyTorch 与 Tensor Core | AMP（自动混合精度）、`torch.float8` 实践 |

---

## 第八篇：大模型推理加速 — 软硬协同全链路（路径 B 核心）

| 章节 | 主题 | 内容 |
|------|------|------|
| 43 | 推理框架概览 | vLLM / SGLang / TensorRT-LLM / LMDeploy 架构对比 |
| 44 | vLLM 深度解析 | PagedAttention、KV Cache 内存管理、Continuous Batching、调度器 |
| 45 | SGLang 深度解析 | RadixAttention、KV Cache 复用、编译器优化、程序化控制 |
| 46 | vLLM vs SGLang 对比 | 设计哲学、性能特征、适用场景选型 |
| 47 | 量化推理 | GPTQ / AWQ / SmoothQuant / FP8（PTQ 全链路），KV Cache 量化 |
| 48 | 投机解码 | Speculative Decoding 原理、Medusa / Eagle / 自回归草稿 |
| 49 | 推理并行策略 | TP/PP/EP 在推理中的部署，通信隐藏，Pipeline 并行气泡消除 |
| 50 | 推理服务化 | 动态批处理、SLO 感知调度、多 LoRA 服务、Speculative Serving |

---

## 第九篇：华为昇腾 CANN 深入（路径 A/B 汇合，重点）

| 章节 | 主题 | 内容 |
|------|------|------|
| 51 | 昇腾硬件架构 | Da Vinci 核心、AICore/AIVector/Cube 单元、HCCS 互连 |
| 52 | CANN 软件栈总览 | GE（图引擎）/ HCCL（通信库）/ ACL（推理接口）/ ATC（模型转换） |
| 53 | CANN Runtime 详解 | Device/Context/Stream/Event 管理、内存分配与生命周期、算子下发管线 |
| 54 | Ascend C 编程 | 编程模型、数据搬运（Data Copy）、Vector/Cube 算子开发 |
| 55 | 自定义算子开发 | TBE / Ascend C 算子注册、编译、集成到 PyTorch |
| 56 | 昇腾上的 PyTorch | `torch_npu` 适配、算子映射、性能调优 |
| 57 | 昇腾推理部署 | ACL 推理接口、ATC 模型转换、OM 模型优化、性能 Profiling |
| 58 | HCCL 通信库 | 集合通信原语、多卡拓扑、与 NCCL 对比 |

---

## 第十篇：实战项目

| 章节 | 主题 | 内容 |
|------|------|------|
| 59 | 项目一 | 手写 QKV 投影 + Flash Attention 算子，PyTorch 集成 benchmark |
| 60 | 项目二 | 基于 vLLM 部署 Qwen2.5 推理服务，Prefill/Decode 性能调优 |
| 61 | 项目三 | 基于 SGLang 构建 Agent 推理服务，KV Cache 复用优化 |
| 62 | 项目四 | 昇腾 NPU 上自定义 Attention 算子 → PyTorch 集成 → 推理部署全链路 |
| 63 | 项目五 | 端到端：Qwen2.5-7B INT8 量化 + FP8 推理 + vLLM 集成，5080 实测 |

---

## 附录

- **数学基础** — 线性代数要点、FP16/BF16/FP8 精度特征与数值稳定性
- **环境搭建** — CUDA Toolkit / cuDNN / PyTorch / CANN 开发环境 Docker 化
- **工具链速查** — nvcc 编译选项、CUDA 版本兼容矩阵、NPU 工具链
- **参考资料** — 经典论文列表、开源项目链接、推荐课程
