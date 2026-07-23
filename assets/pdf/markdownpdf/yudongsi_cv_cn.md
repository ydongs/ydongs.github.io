<img src="https://ydongs.github.io/assets/img/prof_pic-800.webp"
      style="position: absolute; top: 0; right: 0;
            width:100px; height:95px;">

## 司玉栋

<span class="icon">&#xe7ca;</span> `1505632943@qq.com`&emsp;&emsp;
<span class="icon">&#xe600;</span> `https://github.com/yudongsi`

### &#xe80c; 教育经历

<div class="entry-title">
    <h3>同济大学 - 硕士 - 集成电路工程专业</h3>
    <p>2019.09 - 2022.03</p>
</div>

<div class="entry-title">
    <h3>南通大学 - 学士 - 电子信息工程专业</h3>
    <p>2014.09 - 2018.03</p>
</div>

### &#xe618; 工作经验

<div alt="entry-title">
    <h3>AI框架工程师（Triton编译器）@英特尔亚太研发有限公司</h3>
    <p>2022.07 - 2025.07</p>
</div>

#### Triton 编译器 XPU 后端开发

- **后端开发**
  - 增加后端launcher对Global Scratch Memory的支持
  - 构建AOT编译工具链(SPIRV->LevelZero->SYCL)，方便底层问题的复现
  - 开发Intel专用Pass: tritonintelgpu-rewrite-stack-ptr，重写上游共享内存栈指针
  - 增加SPIRV拓展\_Z25\_\_spirv_RoundFToTF32INTELf的支持，提升tl.dot DPAS的精度.
- **性能调优**
  - 建立首套Triton-XPU性能Benchmark体系, 覆盖Softmax, GEMM, FA等流行内核，支持高性能库（如XeTLA, CUTLASS, oneDNN）参考对标
  - 性能优化与提升，支持多种Optimization Pass 落地，包括Coalesce，AcceletateMatmul, RemoveLayoutConversion, Pipeline / Prefetch, Swizzling等
  - 关键内核(如GEMM)性能达到Intel性能库XeTLA的90%+
- **BUG修复**
  - 累计解决编译器后端 35+ High issue， 包括性能回归，Laylout修改传播缺陷，单元测试问题等.

#### PyTorch 生态优化

- **CI/CD 创新**
  - 针对 PyTorch Inductor CPU Performance 基于 Jenkins 设计 AWS Xeon 实例自动化流水线，实现自动收集150+ 模型的性能指标

### &#xecfa; 项目经验

#### 嵌入式计算机视觉系统

- **驾驶员疲劳检测系统 (基于Jetson Nano和IMX6UL开发板)**
  - 针对眼、口和头部姿态，设计了疲劳检测系统，工作流程包括人脸检测、关键点定位、疲劳状态检测和报警
  - 人脸检测算法: HOG+SVM
  - 关键点定位: 利用级联回归树思想训练生成的人脸26点模型
  - 疲劳状态检测: 运行PERCLOS算法和欧拉角计算，识别眯眼、哈欠、低头特征
  - 报警：基于QT C++框架开发操作界面, 适配车载LIN氛围灯提示功能

### &#xecfa; 专业技能

- **编译器**：Triton,LLVM,MLIR,SPIRV
- **异构计算**：oneAPI,OpenCL,CUDA,SYCL,Level Zero,ROCm
- **AI框架**：Pytorch,IPEX
- **AI基础设施**：Jenkins,GitHub Actions,Docker,Vtune Profiler
- **语言**：C/C++,Python,Bash
