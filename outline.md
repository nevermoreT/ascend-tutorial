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

## 第二篇：CUDA 编程基础（路径 A 核心）

| 章节 | 主题 | 内容 |
|------|------|------|
| 06 | CUDA 编程模型 | Grid/Block/Thread 层次，Warp 调度机制，线程束分化 |
| 07 | CUDA 内存体系 | Global/Shared/Constant/Texture/Register，缓存层次，合并访问模式 |
| 08 | CUDA 同步与原子操作 | `__syncthreads()`、原子函数、Warp 级原语（`__shfl_sync`） |
| 09 | CUDA Stream 与 Event | 多流并发、Event 计时、流水线隐藏延迟 |
| 10 | CUDA 错误处理与调试 | `cudaGetLastError`、`cuda-memcheck`、compute sanitizer |

---

## 第三篇：CUDA 进阶优化（路径 B 核心）

| 章节 | 主题 | 内容 |
|------|------|------|
| 11 | 性能 Profiling | Nsight Systems / Nsight Compute 实战，Roofline Model 瓶颈定位 |
| 12 | Tensor Core 编程 | WMMA API、MMA PTX 级操作、混合精度（FP16/BF16/FP8/INT8） |
| 13 | CUDA 协作组与异步 | Cooperative Groups、CUDA Graph、Async Copy（`cp.async`） |
| 14 | 多 GPU 编程 | NCCL 通信原语、NVLink/PCIe 拓扑感知、张量并行通信模式 |

---

## 第四篇：算子开发实战（路径 A/B 分流 → 汇合）

| 章节 | 主题 | 内容 | 路径 |
|------|------|------|------|
| 15 | 算子设计方法论 | 计算密集 vs 内存密集，Roofline 分析性能上限 | A+B |
| 16 | Elementwise 算子 | 向量运算、激活函数（ReLU/GELU/SiLU），kernel fuse | A+B |
| 17 | Reduction 算子 | Warp Reduce → Block Reduce → Grid Reduce，避免 bank conflict | A+B |
| 18 | GEMM 算子（上） | 分块 Tiling 策略、Shared Memory 双缓冲 | B |
| 19 | GEMM 算子（下） | 寄存器分块、Epilogue fusion、Tensor Core 加速 | B |
| 20 | 卷积算子 | im2col + GEMM、Winograd、Implicit GEMM 直接卷积 | B |
| 21 | Flash Attention | 分块策略、Online Softmax、前向与反向传播实现 | B |
| 22 | MoE 相关算子 | Token 路由、All-to-All 通信、Expert 并行 | B |

---

## 第五篇：算子开发框架（路径 B）

| 章节 | 主题 | 内容 |
|------|------|------|
| 23 | Triton 编程 | Block 编程范式、与 CUDA 对比，PyTorch `torch.compile` 集成 |
| 24 | CUTLASS 库 | 模板化 GEMM 架构、Epilogue fusion、混合精度管线 |
| 25 | cuDNN / cuBLAS / TransformerEngine | 高性能库使用与调优 |
| 26 | 算子自动生成简介 | TVM / Tensor Expression / Ansor 自动调优 |

---

## 第六篇：PyTorch 异构计算集成

| 章节 | 主题 | 内容 |
|------|------|------|
| 27 | PyTorch CUDA 扩展 | `torch.utils.cpp_extension`，自定义 CUDA op 注册 |
| 28 | PyTorch `torch.compile` | TorchInductor 后端、自定义后端接入 |
| 29 | PyTorch 分布式训练 | DDP / FSDP 原理，通信与计算重叠 |
| 30 | PyTorch 与 Tensor Core | AMP（自动混合精度）、`torch.float8` 实践 |

---

## 第七篇：大模型推理加速（路径 B 核心）

| 章节 | 主题 | 内容 |
|------|------|------|
| 31 | 推理框架概览 | vLLM / SGLang / TensorRT-LLM / LMDeploy 架构对比 |
| 32 | vLLM 深度解析 | PagedAttention、KV Cache 内存管理、Continuous Batching、调度器 |
| 33 | SGLang 深度解析 | RadixAttention、KV Cache 复用、编译器优化、程序化控制 |
| 34 | vLLM vs SGLang 对比 | 设计哲学、性能特征、适用场景选型 |
| 35 | 量化推理 | GPTQ / AWQ / SmoothQuant / FP8（PTQ 全链路） |
| 36 | 投机解码 | Speculative Decoding 原理、Medusa / Eagle / 自回归草稿 |
| 37 | 推理并行策略 | TP/PP/EP 在推理中的部署，通信隐藏，Pipeline 并行气泡消除 |
| 38 | 推理服务化 | 动态批处理、SLO 感知调度、多 LoRA 服务、Speculative Serving |

---

## 第八篇：华为昇腾 CANN 深入（路径 A/B 汇合，重点）

| 章节 | 主题 | 内容 |
|------|------|------|
| 39 | 昇腾硬件架构 | Da Vinci 核心、AICore/AIVector/Cube 单元、HCCS 互连 |
| 40 | CANN 软件栈总览 | GE（图引擎）/ HCCL（通信库）/ ACL（推理接口）/ ATC（模型转换） |
| 41 | CANN Runtime 详解 | Device/Context/Stream/Event 管理、内存分配与生命周期、算子下发管线 |
| 42 | Ascend C 编程 | 编程模型、数据搬运（Data Copy）、Vector/Cube 算子开发 |
| 43 | 自定义算子开发 | TBE / Ascend C 算子注册、编译、集成到 PyTorch |
| 44 | 昇腾上的 PyTorch | `torch_npu` 适配、算子映射、性能调优 |
| 45 | 昇腾推理部署 | ACL 推理接口、ATC 模型转换、OM 模型优化、性能 Profiling |
| 46 | HCCL 通信库 | 集合通信原语、多卡拓扑、与 NCCL 对比 |

---

## 第九篇：实战项目

| 章节 | 主题 | 内容 |
|------|------|------|
| 47 | 项目一 | 手写 CUDA GEMM + Flash Attention 算子，PyTorch 集成 benchmark |
| 48 | 项目二 | 基于 vLLM 部署 LLM 推理服务，性能调优实战 |
| 49 | 项目三 | 基于 SGLang 构建 Agent 推理服务，KV Cache 复用优化 |
| 50 | 项目四 | 昇腾 NPU 上自定义算子 → PyTorch 集成 → 推理部署全链路 |

---

## 附录

- **数学基础** — 线性代数要点、FP16/BF16/FP8 精度特征与数值稳定性
- **环境搭建** — CUDA Toolkit / cuDNN / PyTorch / CANN 开发环境 Docker 化
- **工具链速查** — nvcc 编译选项、CUDA 版本兼容矩阵、NPU 工具链
- **参考资料** — 经典论文列表、开源项目链接、推荐课程
