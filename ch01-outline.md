# 第一章：GPU 发展历程 — 从图形加速器到 AI 算力核心

> **核心问题**：为什么 GPU 能做 AI 计算，而不是 CPU？
>
> **本章目标**：建立全局认知，理解 GPU 从"显卡"到"AI 引擎"的 25 年演进逻辑

---

## 1.1 什么是 GPU：GPU vs CPU 架构对比

**要点**（~800 字）：
- CPU：少量强核（4~128 核），大面积用于控制逻辑（分支预测、乱序执行、大缓存），追求**低延迟**
- GPU：海量精简核（数千~数万），80%+ 面积用于 ALU，追求**高吞吐**
- 一张架构对比图说明：CPU 像几个教授解复杂题，GPU 像几千个小学生做简单乘法
- 关键数据：A100 有 6912 CUDA Cores，CPU 通常 <100 核；GPU 显存带宽 2~8 TB/s vs CPU DDR5 ~100 GB/s

**参考资料**：
- [Intel 官方：CPU vs GPU 对比](https://www.intel.cn/content/www/cn/zh/products/docs/processors/cpu-vs-gpu.html)
- [阿里云：一文读懂为什么GPU比CPU更快](https://developer.aliyun.com/article/1391088)
- [腾讯云：CPU与GPU的算力演进](https://cloud.tencent.com/developer/article/2556778)
- [John's Blog：CPU&GPU架构差异及AI场景中的应用](https://johng.cn/ai/cpu-gpu)

---

## 1.2 图形时代（1999-2005）：固定管线 → 可编程着色器

**要点**（~600 字）：
- **1999 GeForce 256**：NVIDIA 首次提出"GPU"概念，集成硬件 T&L（Transform & Lighting），固定功能渲染管线
- **2001 GeForce 3**：DirectX 8 引入可编程 Vertex Shader + Pixel Shader，GPU 开始"可编程"
- **2002-2005**：着色器程序从 4 条指令增长到数百条，GPU 算力增速 2.4x/年（远超摩尔定律 1.8x/年）
- 为什么图形渲染天然并行？——屏幕上百万像素，每个像素独立计算颜色

**参考资料**：
- [NVIDIA Research: Evolution of the GPU](https://research.nvidia.com/publication/2021-12_evolution-graphics-processing-unit-gpu) — NVIDIA 官方 GPU 演进综述论文
- [GPU History Paper (Emory University)](http://www.cs.emory.edu/~cheung/Courses/355/Syllabus/94-CUDA/Docs/gpu-hist-paper.pdf) — 从固定管线到统一着色器的架构演进
- [微信公众号：从计算机图形学发展简史看GPU演进](https://mp.weixin.qq.com/s/kDSTaFdpnssD_I3tBqfN-A)
- [廖雪峰：GPU简史兼论Nvidia的崛起](https://liaoxuefeng.com/blogs/all/2024-06-21-gpu/index.html)

---

## 1.3 GPGPU 先驱与 CUDA 诞生（2006-2010）

**要点**（~800 字）：
- **早期 GPGPU（2002-2005）**：科学家用 OpenGL 着色器做通用计算（把数据伪装成纹理），极其痛苦
- **2004 Brook GPU**：Stanford 博士生 Ian Buck（导师 Pat Hanrahan）开发的流编程语言，CUDA 的前身
- **2006 G80（GeForce 8800）**：首个统一着色器架构（Unified Shader），128 个流处理器，345.6 GFLOPS
- **2007 CUDA 1.0**：Ian Buck 加入 NVIDIA 后打造，C 语言直接编程 GPU，绕过图形 API
- **2010 Fermi（GF100）**：首个专为 GPGPU 设计的 GPU — 真正的缓存层次、ECC、C++ 支持
- 为什么 CUDA 赢了 OpenCL？——生态投入、硬件耦合、库生态（cuBLAS/cuDNN）

**参考资料**：
- [NVIDIA Blog: Inside the Programming Evolution of GPU Computing](https://developer.nvidia.com/blog/inside-the-programming-evolution-of-gpu-computing/) — Ian Buck 亲述 CUDA 诞生
- [NVIDIA Blog: CUDA Refresher - Origins of GPU Computing](https://developer.nvidia.com/blog/cuda-refresher-reviewing-the-origins-of-gpu-computing/) — 图形性能增长曲线、延迟隐藏原理
- [Wikipedia: CUDA](https://en.wikipedia.org/wiki/CUDA) — CUDA 历史、Ian Buck 与 Brook
- [The CUDA Handbook: Ten Years Later - Why CUDA Succeeded](https://www.cudahandbook.com/2017/06/ten-years-later-why-cuda-succeeded/) — CUDA 成功原因深度分析
- [Beyond3D: NVIDIA Tesla - GPU computing gets its own brand](https://www.beyond3d.com/content/articles/77/1) — G80/CUDA 发布时的深度报道
- [NVIDIA Fermi Whitepaper](https://www.nvidia.com/content/PDF/fermi_white_papers/N.Brookwood_NVIDIA_Solves_the_GPU_Computing_Puzzle1.pdf)
- [IEEE Micro: GPU Architecture (2010)](https://pages.cs.wisc.edu/~markhill/restricted/ieeemicro10_gpu.pdf) — Fermi 架构学术分析

---

## 1.4 从 Fermi 到 Volta：算力飞升（2010-2017）

**要点**（~800 字，配合规格表）：

| 架构 | 年份 | 代表产品 | CUDA Cores | 关键创新 | FP32 TFLOPS |
|------|------|---------|-----------|---------|-------------|
| Fermi | 2010 | GTX 480 | 512 | 首个 GPGPU GPU，缓存+ECC | 1.6 |
| Kepler | 2012 | GTX 680 / K40 | 2880 | Dynamic Parallelism, Hyper-Q | 4.3 |
| Maxwell | 2014 | GTX 980 | 2048 | 能效优化，统一内存 | 4.6 |
| Pascal | 2016 | P100 / GTX 1080 | 3584 | HBM2, NVLink 1.0, 原生 FP16 | 10.6 |
| Volta | 2017 | V100 | 5120 | **Tensor Core 诞生**, NVLink 2.0 | 15.7 (FP32) / 125 (Tensor FP16) |

- Pascal P100：首款被大规模用于深度学习的数据中心 GPU
- Volta V100：Tensor Core 的问世——矩阵乘法硬件化，FP16 算力从 15.7T → 125T（8x 提升）

**参考资料**：
- [ProcessorHistory: GPU History - TFLOPS, VRAM & Transistor Growth 1999-2025](https://processorhistory.com/gpu-history/) — 算力/晶体管/显存增长曲线
- [noze.it: NVIDIA GPU architectures from Tesla to Blackwell](https://www.noze.it/en/insights/nvidia-gpu-architectures/) — 每代架构关键特性总结
- [NVIDIA GTC 2010: The Evolution of GPUs for GPGPU](https://www.nvidia.com/content/gtc-2010/pdfs/2275_gtc2010.pdf) — 早期 GPU 计算演进 PPT
- [Wikipedia: Ampere microarchitecture](https://en.wikipedia.org/wiki/Ampere_(microarchitecture)) — 含 V100/H100/B100 详细规格表
- [CSDN：万字长文 英伟达GPU硬件架构发展史全景回顾](https://blog.csdn.net/weixin_43719763/article/details/150474359)
- [CSDN：GPU/CUDA 发展编年史](https://blog.csdn.net/Jmilk/article/details/145970533)

---

## 1.5 Tensor Core 时代与 AI 爆发（2012-2024）

**要点**（~800 字）：
- **2012 AlexNet 时刻**：Alex Krizhevsky 用 2 块 GTX 580 训练 CNN，ImageNet 错误率下降 10 个百分点，深度学习革命开始
- **Volta（2017）**：1st Gen Tensor Core — 4×4 矩阵乘累加（HMMA），FP16 only
- **Ampere A100（2020）**：3rd Gen Tensor Core — 引入 BF16、TF32、2:4 稀疏、MIG
- **Hopper H100（2022）**：4th Gen Tensor Core + Transformer Engine — FP8 支持，动态精度管理
- **Blackwell B200（2024）**：5th Gen Tensor Core — FP4/FP6 原生支持，双芯互联（NV-HBI 10 TB/s），192GB HBM3e

算力增长对比（Tensor Core FP16/BF16）：
```
V100 (2017):    125 TFLOPS
A100 (2020):    624 TFLOPS   (5x)
H100 (2022):  2,000 TFLOPS   (16x vs V100)
B200 (2024):  4,500 TFLOPS   (36x vs V100)
```

**参考资料**：
- [SemiAnalysis: NVIDIA Tensor Core Evolution from Volta to Blackwell](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell) — **必读**，Tensor Core 每代架构深度分析
- [Exxact: Blackwell vs Hopper GPU Comparison](https://www.exxactcorp.com/blog/hpc/comparing-nvidia-tensor-core-gpus) — B200/H200/H100/A100 规格对比表
- [IntuitionLabs: Blackwell vs Hopper Architecture Comparison](http://intuitionlabs.ai/articles/blackwell-vs-hopper-gpu-architecture-comparison) — 架构差异深度对比
- [Wikipedia: Blackwell microarchitecture](https://en.wikipedia.org/wiki/Blackwell_(microarchitecture)) — RTX 50 系列规格（含 5080 的 GB203: 10752 CUDA, 336 TC, 16GB GDDR7）
- [AlexNet 原始论文](https://web.stanford.edu/class/cs231m/references/alexnet.pdf)
- [Pinecone: AlexNet and ImageNet - The Birth of Deep Learning](https://www.pinecone.io/learn/series/image-search/imagenet/)
- [arXiv: GPU Landscape for AI (2025)](https://arxiv.org/pdf/2601.20115) — 最新 GPU 架构综述论文

---

## 1.6 AI 芯片格局：GPU / TPU / NPU 定位

**要点**（~600 字，简略，后续章节展开）：
- **GPU**：通用性最强，训练+推理通吃，生态最完善（CUDA + PyTorch）
- **TPU**：Google 专用，脉动阵列，大规模训练极致性价比，但生态封闭
- **NPU**：华为昇腾（Da Vinci）、寒武纪，国产替代方向，推理场景优势
- 训练芯片 vs 推理芯片的设计权衡：算力 vs 带宽 vs 功耗 vs 灵活性
- 一张格局图：不同芯片在"灵活性-效率"谱系上的位置

**参考资料**：
- [腾讯网：从Tesla到Blackwell，英伟达如何改写HPC规则](https://news.qq.com/rain/a/20250318A07GUD00)
- [百度云：从图形渲染到通用计算 GPU技术演进](https://cloud.baidu.com/article/4618712)
- [38oo.com: NVIDIA GPU架构演进全解析](https://www.38oo.com/wenzhang/6f36e550-4146-11f1-878f-f1b484b05fd0)

---

## 1.7 本章小结

- GPU 用 25 年完成了"图形专用 → 通用并行 → AI 核心"的三级跳
- 三次关键转折：① 统一着色器（G80）② CUDA（2007）③ Tensor Core（Volta 2017）
- 下一章预告：深入 CUDA 编程模型，理解 Grid/Block/Thread 层次

---

## 附录：关键 NVIDIA 架构代号速查

| 代号 | 架构名 | 年份 | 消费级产品 | 数据中心产品 |
|------|--------|------|-----------|-------------|
| NV4x | Celsius | 1999 | GeForce 256/2 | - |
| NV20 | Kelvin | 2001 | GeForce 3/4 | - |
| NV30 | Rankine | 2003 | GeForce 5 | - |
| NV40 | Curie | 2004 | GeForce 6/7 | - |
| G80 | Tesla | 2006 | GeForce 8/9/200 | Tesla C870 |
| GF100 | Fermi | 2010 | GeForce 400/500 | Tesla C2050 |
| GK110 | Kepler | 2012 | GeForce 600/700 | Tesla K40 |
| GM200 | Maxwell | 2014 | GeForce 700/900 | Tesla M40 |
| GP100 | Pascal | 2016 | GeForce 10 | Tesla P100 |
| GV100 | Volta | 2017 | Titan V | Tesla V100 |
| TU102 | Turing | 2018 | GeForce RTX 20 | Quadro RTX |
| GA100 | Ampere | 2020 | GeForce 30 | A100 |
| AD102 | Ada Lovelace | 2022 | GeForce 40 | L40S |
| GH100 | Hopper | 2022 | - | H100/H200 |
| GB202 | Blackwell | 2024/25 | GeForce 50 | B100/B200 |

> 来源：[Wikipedia: Category:Nvidia microarchitectures](https://en.m.wikipedia.org/wiki/Category:Nvidia_microarchitectures)
