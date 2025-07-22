<img src="https://ydongs.github.io/assets/img/prof_pic-800.webp"
      style="position: absolute; top: 0; right: 0;
            width:100px; height:95px;">

## 司玉栋

<span class="icon">&#xe60f;</span> `19121726080`&emsp;&emsp;
<span class="icon">&#xe7ca;</span> `1505632943@qq.com`&emsp;&emsp;
<span class="icon">&#xe600;</span> `https://github.com/yudongsi`

### &#xe80c; 教育经历

<div class="entry-title">
    <h3>同济大学 - 硕士 - 集成电路工程专业</h3>
    <p>2019.09 - 2022.03</p>
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
  - 性能优化与提升，关键内核(如GEMM)性能达到Intel性能库XeTLA的90%+
- **BUG修复**
  - 累计解决编译器后端 35+ High issue， 包括性能回归，Laylout修改传播缺陷，单元测试问题等.

#### PyTorch 生态优化

- **CI/CD 创新**
  - 针对 PyTorch Inductor CPU Performance 基于 Jenkins 设计 AWS Xeon 实例自动化流水线，实现：
    - 自动收集200+ 模型的性能指标
    - 自动化报告生成与发布

<div class="entry-title">
  <h3>CUDA测开实习生@英伟达半导体科技（上海）有限公司</h3>
  <p>2021.08 - 2021.11</p>
</div>

#### CUDA Orin Simulator实践

- **参与NVIDIA Orin t23x SOC的CUDA安全与代码覆盖测试体系构建**
  - 基于VDK虚拟测试套件搭建自动化测试平台
  - 定位failed/timeout用例
  - 实现测试镜像版本自动化追踪

### &#xecfa; 专业技能

- **编译器**：Triton,LLVM,MLIR,SPIRV
- **异构计算**：oneAPI,OpenCL,CUDA,SYCL,Level Zero,ROCm
- **AI框架**：Pytorch,IPEX
- **AI基础设施**：Jenkins,GitHub Actions,Docker,Vtune Profiler
- **语言**：C/C++,Python,Bash
