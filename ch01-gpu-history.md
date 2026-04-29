# 第一章：GPU 发展历程 — 从图形加速器到 AI 算力核心

> **核心问题**：为什么 GPU 能做 AI 计算，而不是 CPU？
>
> **本章目标**：建立全局认知，理解 GPU 从"显卡"到"AI 引擎"的 25 年演进逻辑

---

## 1.1 什么是 GPU：GPU vs CPU 架构对比

要理解 GPU，先从它和 CPU 的根本区别说起。

CPU（Central Processing Unit）和 GPU（Graphics Processing Unit）都是处理器，但设计哲学截然不同。CPU 追求**低延迟**——尽快完成一个任务；GPU 追求**高吞吐**——同时处理尽可能多的任务。

打个比方：CPU 像几个顶级教授，能解极其复杂的微积分题，但人数少；GPU 像几千个小学生，只会做加减乘除，但可以同时开工。如果你需要做一千万次矩阵乘法，选谁不言而喻。

这种差异直接体现在芯片面积的分配上：

```
CPU 芯片面积分配：                    GPU 芯片面积分配：
┌───────────────────────┐           ┌───────────────────────┐
│                       │           │  ALU  ALU  ALU  ALU   │
│   控制逻辑 (~40%)     │           │  ALU  ALU  ALU  ALU   │
│   分支预测/乱序执行    │           │  ALU  ALU  ALU  ALU   │
│                       │           │  ALU  ALU  ALU  ALU   │
│   L1/L2/L3 Cache      │           │  ALU  ALU  ALU  ALU   │
│   (~30%)              │           │  ALU  ALU  ALU  ALU   │
│                       │           │                       │
│   ALU (~15%)          │           │  控制逻辑 + Cache     │
│                       │           │  (~20%)               │
└───────────────────────┘           └───────────────────────┘

CPU: 大量面积给控制和缓存，ALU 占比小
GPU: 80%+ 面积给 ALU，控制和缓存精简
```

用实际数据说话：

| 指标 | CPU（Intel Xeon w9-3495X） | GPU（NVIDIA H100） |
|------|---------------------------|-------------------|
| 核心数 | 56 核 | 16,896 CUDA Cores + 528 Tensor Cores |
| 主频 | 最高 4.8 GHz | ~1.98 GHz |
| 内存带宽 | ~200 GB/s（DDR5） | 3,350 GB/s（HBM3） |
| FP32 算力 | ~数 TFLOPS | 67 TFLOPS（标准）/ 989 TFLOPS（Tensor Core） |
| 设计目标 | 低延迟、通用性 | 高吞吐、数据并行 |

CPU 的 56 个核每个都是"全功能"的——有独立的分支预测器、乱序执行引擎、深层流水线。GPU 的 16,896 个核心则极其精简，没有复杂的控制逻辑，但它们可以通过一种叫"延迟隐藏"（Latency Hiding）的技巧弥补：当某组线程在等内存数据时，硬件瞬间切换到另一组线程继续计算，保持 ALU 永远不空闲。

**一句话总结**：CPU 是"万能瑞士军刀"，什么都能干但并行有限；GPU 是"大规模并行计算专用引擎"，在数据并行场景下拥有不可替代的优势。而深度学习——本质上是海量矩阵乘法——恰好是最典型的数据并行场景。

---

## 1.2 图形时代（1999-2005）：固定管线到可编程着色器

GPU 的诞生源于一个朴素的需求：3D 游戏画面太卡了。

### 固定功能管线（1999-2001）

1999 年，NVIDIA 发布 GeForce 256，第一次提出"GPU"（Graphics Processing Unit）这个名字。在这之前，PC 的 3D 图形渲染主要靠 CPU 硬算，帧率可怜得可以——跑《雷神之锤 3》能稳 20 帧就算不错了。

GeForce 256 的核心创新是把 **T&L（Transform & Lighting，几何变换与光照计算）**从 CPU 卸载到了专用硬件。渲染管线变成了这样的固定流水线：

```
顶点数据 → [几何变换(T)] → [光照计算(L)] → [三角形setup] → [像素着色] → 显示器
              ↑ 硬件固化，不可编程
```

这叫**固定功能管线**（Fixed-Function Pipeline）——每个阶段的计算逻辑被焊死在硬件里，开发者无法修改。虽然不灵活，但胜在快。

### 可编程着色器（2001-2005）

2001 年，微软 DirectX 8 引入了可编程的 Vertex Shader（顶点着色器）和 Pixel Shader（像素着色器）。同年 NVIDIA 的 GeForce 3 成为首款支持可编程着色器的消费级 GPU。

这意味着开发者终于可以自己写小程序来控制顶点变换和像素颜色——GPU 开始"可编程"了。着色器程序从最初只有 4 条指令，迅速增长到数百条。

一个关键数据：从 1999 到 2005 年，GPU 图形性能增长速度约为 **2.4 倍/年**，远超摩尔定律预测的 1.8 倍/年。这种超常规增速的原因在于图形渲染天然是大规模并行的——屏幕上有百万个像素，每个像素的颜色可以独立计算，完美契合硬件并行。

然而，这一时期的 GPU 仍然只为图形服务。要让它做科学计算？你得把计算任务"伪装"成纹理贴图和像素着色操作——极其痛苦，但已经有人在这么做了。

---

## 1.3 GPGPU 先驱与 CUDA 诞生（2006-2010）

### 早期的 GPGPU 黑客（2002-2005）

在 CUDA 诞生之前，一小撮科学家和工程师已经在"滥用"GPU 做通用计算了。

方法很 hack：把待计算的数据编码成图片纹理（texture），把计算逻辑写成着色器程序（shader），然后把结果当作渲染后的图片读回来。GPU 算完后输出的不是画面，而是一堆浮点数。

这个过程极其别扭。你需要精通 OpenGL 或 Direct3D 图形 API，需要理解纹理坐标和帧缓冲，还要处理 GPU 不支持的散列（scatter）操作。但即使如此，性能提升也足够诱人——在某些科学计算场景下，GPU 比 CPU 快 10-30 倍。

Stanford 大学的一批人决定改变这种局面。

### Brook 与 CUDA 的诞生

2004 年，Stanford 博士生 **Ian Buck**（导师是 2019 年图灵奖得主 Pat Hanrahan）发表了论文《Brook for GPUs: Stream Computing on Graphics Hardware》。Brook 是一种流编程语言，基于 C 语言扩展，让开发者可以直接表达数据并行计算，而不用去折腾图形 API。Brook 可以在 GeForce 6 系列上实现矩阵运算、FFT 等通用算法。

NVIDIA 看到了这项工作的潜力。2004 年，NVIDIA 把 Ian Buck 招入麾下，让他与首席架构师 John Nickolls 一起，将 Brook 演化为一个全新的并行计算平台。

### 2006：G80 — 统一着色器 + CUDA

2006 年 11 月，NVIDIA 发布 **GeForce 8800 GTX**（G80 架构），这是 GPU 历史上最重要的转折点之一。

G80 做了两件划时代的事：

**1. 统一着色器架构（Unified Shader）**

之前的 GPU 把顶点着色器和像素着色器做成独立的硬件单元。如果一个游戏场景顶点多、像素少，顶点单元忙死、像素单元闲死。G80 打破了这个界限——所有着色器统一为**流处理器（Streaming Processor, SP）**，可以动态分配给任何类型的计算任务。

```
之前：  [Vertex Shader × N] [Pixel Shader × M]  ← 固定比例，利用率低
G80：   [Unified SP × 128]                        ← 统一调度，利用率高
```

这 128 个 SP 就是后来"CUDA Core"的前身。G80 还引入了 **SM（Streaming Multiprocessor）** 的概念——每 8 个 SP 组成一个 SM，作为基本的调度单元。这个 SM 的概念一直延续到今天。

**2. CUDA 1.0（2007 年正式发布）**

CUDA（Compute Unified Device Architecture）让开发者用标准 C 语言直接编程 GPU，不再需要经过图形 API。

CUDA 定义了一套清晰编程抽象：

```
Grid（网格）
 └── Block（线程块）
      └── Thread（线程）
```

- **Thread**：最小执行单元，每个线程处理一个数据点
- **Block**：一组线程，共享 fast shared memory，可以同步
- **Grid**：所有 Block 的集合，代表一次 kernel launch

这个模型简单到令人惊讶——它把 GPU 的复杂硬件结构抽象成了开发者熟悉的层次。一个 C 程序员加几行关键字就能在 GPU 上跑并行代码。

GeForce 8800 GTX 的规格：128 个 SP，345.6 GFLOPS FP32，86.4 GB/s 显存带宽。同期的高端 CPU 算力约为 50 GFLOPS——GPU 已经拉开了接近 7 倍的差距。

### 2010：Fermi — 首个真正的 GPGPU 架构

如果说 G80 是"顺带支持计算"，那 2010 年的 **Fermi（GF100）** 就是第一个从根基上为通用计算设计的 GPU。

Fermi 带来了：
- **真正的缓存层次**：L1 Cache（每个 SM 64KB）+ 统一 L2 Cache（768KB），之前的 GPU 几乎没有缓存
- **ECC 内存**：科学计算对数据正确性要求极高，ECC 纠错首次上 GPU
- **C++ 支持**：CUDA 不再局限于 C 子集
- 512 个 CUDA Cores，组成 16 个 SM

Fermi 的意义在于：它让 GPU 从"游戏卡的副产品"变成了一个正经的高性能计算设备。此后，GPU 不再只是"显卡"，而是一个通用的并行计算加速器。

### 为什么 CUDA 赢了 OpenCL？

CUDA 并非唯一选择。Apple 和 Khronos Group 推出了 OpenCL（2008），目标是跨厂商的开放标准——不只是 NVIDIA，AMD 和 Intel 的 GPU 也能用。

但 CUDA 最终胜出，原因有三：

1. **硬件-软件深度耦合**：NVIDIA 每一代新架构都伴随 CUDA 新特性（Tensor Core、Cooperative Groups、CUDA Graph），OpenCL 作为跨厂商标准，永远只能取最小公倍数
2. **库生态**：cuBLAS（2007）、cuDNN（2014）——深度学习框架的底层几乎全部基于这些库
3. **持续投入**：NVIDIA 把 CUDA 当核心战略，投入了超过十年的工程师资源。OpenCL 缺乏这种级别的单一公司推动

到 2015 年左右，深度学习社区已经形成"用 GPU = 用 NVIDIA = 用 CUDA"的事实标准。这个生态壁垒至今无人打破。

---

## 1.4 从 Fermi 到 Volta：算力飞升（2010-2017）

2010 年到 2017 年间，GPU 架构经历了五代演进。每一代都在三个方向上推进：更多核心、更高带宽、更好可编程性。

### 五代架构规格一览

| 架构 | 年份 | 代表产品 | CUDA Cores | FP32 TFLOPS | 关键创新 |
|------|------|---------|-----------|-------------|---------|
| **Fermi** | 2010 | GTX 480 | 512 | 1.6 | 缓存层次、ECC、C++ |
| **Kepler** | 2012 | GTX 680 / K40 | 2,880 | 4.3 | Dynamic Parallelism、Hyper-Q |
| **Maxwell** | 2014 | GTX 980 / M40 | 2,048 | 4.6 | 极致能效比、统一内存 |
| **Pascal** | 2016 | P100 / GTX 1080 | 3,584 | 10.6 | **HBM2**、**NVLink 1.0**、原生 FP16 |
| **Volta** | 2017 | V100 | 5,120 | 15.7 | **Tensor Core**、NVLink 2.0 |

> 注：FLOPS = FLoating-point Operations Per Second。1 TFLOPS = 每秒 1 万亿次浮点运算。

### 几个关键里程碑

**Kepler（2012）——Dynamic Parallelism**

Kepler 允许 GPU kernel 自己在 GPU 上启动新的 kernel，不必回 CPU。这减少了 CPU-GPU 之间的通信往返，让 GPU 更加"自治"。Kepler 也是第一款 SM 内含 192 个 CUDA Core 的架构（Fermi 只有 32 个），单 SM 性能跃升。

**Maxwell（2014）——能效革命**

Maxwell 没有追求核心数量的增长（甚至比 Kepler 少），而是重新设计 SM 内部结构，大幅提升每个核心的利用率。结果是在相同的功耗下，性能提升了 2 倍。这证明了一个道理：**盲目堆核心不如优化架构**。

**Pascal（2016）——HBM2 与 NVLink**

Pascal P100 是第一款使用 **HBM2（High Bandwidth Memory）** 的 NVIDIA GPU。HBM2 通过在 GPU 芯片旁边堆叠内存芯片，提供 720 GB/s 的带宽——是当时 GDDR5 的 3 倍。对于深度学习这种内存密集型负载，带宽直接制约性能上限。

Pascal 同时引入了 **NVLink 1.0**（160 GB/s），用于多 GPU 之间的高速直连，绕过 PCIe 的带宽瓶颈。

但 Pascal 最重要的贡献可能是**让深度学习变得实用**。Tesla P100 是第一款被大规模部署于深度学习训练集群的数据中心 GPU。2016 年的百度、Google、Facebook 都在大规模采购 P100。

**Volta（2017）——Tensor Core 诞生**

Volta V100 改变了一切。

NVIDIA 发现，深度学习的核心计算是矩阵乘法（GEMM），而标准的 CUDA Core 做矩阵乘法效率不高——每条指令只能完成一次乘加（FMA），指令调度开销占比太大。

解决方案是 **Tensor Core**——一种专门做矩阵乘累加（MMA）的硬件单元。每个 Tensor Core 可以在一个时钟周期内完成一个 **4×4 矩阵的乘累加**操作：

```
D[4×4] = A[4×4] × B[4×4] + C[4×4]
```

V100 有 640 个 Tensor Core，FP16 矩阵乘算力达到 **125 TFLOPS**——是 FP32 标准算力的 8 倍。

Tensor Core 的引入并非长期规划。据 SemiAnalysis 报道，Tensor Core 是在 Volta 流片前几个月才临时加入的——NVIDIA 敏锐捕捉到深度学习对矩阵计算的爆发式需求，以惊人的速度完成了架构调整。

---

## 1.5 Tensor Core 时代与 AI 爆发（2012-2024）

### 2012：AlexNet — 深度学习的"大爆炸"

2012 年 9 月 30 日，ImageNet 图像识别挑战赛（ILSVRC 2012）的结果震惊了学术界。

Alex Krizhevsky 设计的 **AlexNet**——一个深度卷积神经网络——以 top-5 错误率 16.4% 夺冠，比第二名低了超过 **10 个百分点**。在 AI 领域，领先 1-2% 就算重大突破，10% 的差距堪称碾压。

AlexNet 的关键：**两块 NVIDIA GTX 580 GPU**。

这个 6000 万参数的网络，用两块消费级显卡训练了 5-6 天。每块 GTX 580 只有 3GB 显存，Alex 把网络拆成两半分别放在两张卡上，只在特定层做跨卡通信——这大概是最早的"模型并行"实践。

AlexNet 证明了两件事：
1. 深度神经网络真的管用——只要足够深、数据足够多
2. GPU 是训练深度网络的唯一实用选择——CPU 太慢了

此后，所有 AI 研究者开始购买 NVIDIA 显卡。NVIDIA 的命运从此改变。

### Tensor Core 四代演进

AlexNet 之后的十年，AI 模型的参数量从千万级（AlexNet）膨胀到万亿级（GPT-4），对算力的渴求几乎无止境。NVIDIA 的回应是 Tensor Core 的持续迭代：

**Volta V100（2017）——1st Gen Tensor Core**

- 仅支持 FP16 输入/FP16 累加
- 每个 Tensor Core：4×4 FP16 矩阵乘累加/时钟周期
- FP16 Tensor 算力：125 TFLOPS
- BERT-Large 训练时间：从 7 天缩短到 47 小时

**Ampere A100（2020）——3rd Gen Tensor Core**

- 新增 **BF16**（Brain Float，更大的指数位，训练更稳定）、**TF32**（TensorFloat-32，无需改代码即可加速 FP32 训练）
- 引入 **2:4 结构化稀疏**——权重中 50% 为零时，算力翻倍
- **MIG（Multi-Instance GPU）**：一张 A100 可切分为最多 7 个独立实例
- FP16 Tensor 算力：624 TFLOPS（5x V100）
- 80GB HBM2e，2 TB/s 带宽

A100 是大模型时代的"标准算力单位"。GPT-3（1750 亿参数）的训练集群就由数千张 A100 组成。

**Hopper H100（2022）——4th Gen Tensor Core + Transformer Engine**

- **FP8** 支持（E4M3 和 E5M2 两种格式），训练和推理的精度/效率新平衡
- **Transformer Engine**：软件层面自动管理 FP8/FP16 混合精度，无需手动量化
- NVLink 4.0：900 GB/s 的 GPU 间通信带宽
- FP16 Tensor 算力：2,000 TFLOPS（16x V100）
- FP8 Tensor 算力：4,000 TFLOPS
- 80GB HBM3，3.35 TB/s 带宽

H100 是 ChatGPT 爆发后最紧俏的硬件——2023 年一张 H100 的黑市价格被炒到数万美元。

**Blackwell B200（2024）——5th Gen Tensor Core**

- **FP4/FP6 原生支持**——推理场景的终极精度压缩
- 双芯通过 **NV-HBI（10 TB/s）** 互联，软件视为单卡
- 第二代 Transformer Engine，动态混合 FP4/FP6/FP8/FP16
- FP16 Tensor 算力：4,500 TFLOPS（36x V100）
- FP4 Tensor 算力：9,000 TFLOPS
- 192GB HBM3e，8 TB/s 带宽
- **消费级的 RTX 5080（你的卡！）** 也使用 Blackwell 架构（GB203），配备 10,752 个 CUDA Cores 和 336 个 5th Gen Tensor Cores

### 算力增长的曲线

```
Tensor Core FP16/BF16 算力增长（TFLOPS）：

V100  (2017)  ▓ 125
A100  (2020)  ▓▓▓▓▓ 624           (5x)
H100  (2022)  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 2,000  (16x vs V100)
B200  (2024)  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 4,500  (36x vs V100)
```

七年时间，算力增长了 36 倍。这远超摩尔定律的预测，其驱动力不是制程进步，而是**专用硬件（Tensor Core）+ 低精度计算（FP16→FP8→FP4）**的架构创新。

---

## 1.6 AI 芯片格局：GPU / TPU / NPU

GPU 是 AI 算力的主力，但不是唯一玩家。从全局视角看，AI 芯片可以分为三大阵营：

### GPU — 通用算力王者

- **代表**：NVIDIA A100 / H100 / B200
- **优势**：最强的通用性（训练+推理+图形+科学计算），最完善的软件生态（CUDA + cuDNN + TensorRT），PyTorch/TensorFlow 一等公民
- **劣势**：价格昂贵（H100 数万美元），功耗高（700W+），存在大量对 AI 无用的图形硬件
- **定位**：AI 训练的默认选择，大规模推理的高性能选择

### TPU — Google 的专用利器

- **代表**：TPU v1~v5
- **优势**：脉动阵列（Systolic Array）针对矩阵乘法极致优化，大规模训练性价比高，与 Google Cloud/JAX 深度集成
- **劣势**：仅限 Google 生态，外部开发者难以使用，推理灵活性不如 GPU
- **定位**：Google 内部大规模训练的首选

### NPU — 国产替代路径

- **代表**：华为昇腾 910B（Da Vinci 架构）、寒武纪思元、苹果 Neural Engine
- **优势**：国产化替代（不受出口管制），特定场景功耗低，推理端性价比优
- **劣势**：软件生态仍在建设中（CANN vs CUDA），训练场景成熟度不足，社区和文档较少
- **定位**：国产推理部署、边缘端 AI、政府/金融等信创场景

### 训练 vs 推理芯片的设计权衡

```
              灵活性高                          专用效率高
          ←——————————————┬——————————————→
           GPU            TPU             NPU/ASIC
         
     训练需要灵活性         推理可以固化
     （动态图、混合精度）    （固定模型、低精度）
```

训练芯片需要支持灵活的计算图、多种精度、动态 shape，GPU 的通用性在这里是优势。推理芯片可以针对特定模型结构做硬化，牺牲灵活性换取极致的效率。

---

## 1.7 本章小结

回顾 GPU 25 年的演进，有三次关键转折：

1. **统一着色器（2006，G80）**：GPU 从多个专用单元变成统一的可编程处理器，为通用计算铺路
2. **CUDA（2007）**：GPU 不再需要"伪装"成图形设备，标准 C 语言直接编程
3. **Tensor Core（2017，Volta）**：矩阵乘法硬件化，AI 算力从此指数级增长

这三次转折的逻辑是一致的：**从专用到通用，再从通用中提炼出新的专用加速**。GPU 起初是图形专用硬件，通过可编程化变成了通用并行处理器，又在 AI 时代演化出 Tensor Core 这样的专用计算单元。

理解了这个演进逻辑，后面学习 CUDA 编程、算子优化、推理加速时，你会知道每一个技术决策背后的"为什么"。

**下一章**：深入 CUDA 编程模型，理解 Grid/Block/Thread 层次结构，写出你的第一个 GPU 程序。

---

## 附录：NVIDIA GPU 架构代号速查表

| 代号 | 架构名 | 年份 | 消费级产品 | 数据中心产品 | 核心创新 |
|------|--------|------|-----------|-------------|---------|
| NV4x | Celsius | 1999 | GeForce 256/2 | — | 首次提出 GPU 概念 |
| NV20 | Kelvin | 2001 | GeForce 3/4 | — | 可编程顶点着色器 |
| NV30 | Rankine | 2003 | GeForce FX 5 | — | 像素着色器 2.0 |
| NV40 | Curie | 2004 | GeForce 6/7 | — | SLI 多卡 |
| G80 | Tesla | 2006 | GeForce 8/9/200 | Tesla C870 | **统一着色器 + CUDA** |
| GF100 | Fermi | 2010 | GeForce 400/500 | Tesla C2050 | 缓存层次、ECC、C++ |
| GK110 | Kepler | 2012 | GeForce 600/700 | Tesla K40 | Dynamic Parallelism |
| GM200 | Maxwell | 2014 | GeForce 700/900 | Tesla M40 | 极效能比 |
| GP100 | Pascal | 2016 | GeForce 10 | Tesla P100 | HBM2、NVLink |
| GV100 | Volta | 2017 | Titan V | Tesla V100 | **Tensor Core** |
| TU102 | Turing | 2018 | GeForce RTX 20 | Quadro RTX | RT Core、DLSS |
| GA100 | Ampere | 2020 | GeForce 30 | A100 | 3rd TC、MIG、稀疏 |
| AD102 | Ada Lovelace | 2022 | GeForce 40 | L40S | 4th TC、SER |
| GH100 | Hopper | 2022 | — | H100/H200 | FP8、Transformer Engine |
| GB202 | Blackwell | 2024/25 | GeForce 50 | B100/B200 | FP4、双芯互联 |

> **扩展阅读**：
> - [NVIDIA Research: Evolution of the GPU](https://research.nvidia.com/publication/2021-12_evolution-graphics-processing-unit-gpu)
> - [SemiAnalysis: NVIDIA Tensor Core Evolution from Volta to Blackwell](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell)
> - [ProcessorHistory: GPU History — TFLOPS, VRAM & Transistor Growth 1999–2025](https://processorhistory.com/gpu-history/)
> - [NVIDIA Blog: CUDA Refresher — Origins of GPU Computing](https://developer.nvidia.com/blog/cuda-refresher-reviewing-the-origins-of-gpu-computing/)
> - [The CUDA Handbook: Ten Years Later — Why CUDA Succeeded](https://www.cudahandbook.com/2017/06/ten-years-later-why-cuda-succeeded/)
> - [AlexNet 原始论文](https://web.stanford.edu/class/cs231m/references/alexnet.pdf)
> - [CSDN：GPU/CUDA 发展编年史](https://blog.csdn.net/Jmilk/article/details/145970533)
> - [廖雪峰：GPU简史兼论Nvidia的崛起](https://liaoxuefeng.com/blogs/all/2024-06-21-gpu/index.html)
