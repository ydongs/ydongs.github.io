<img src="https://ydongs.github.io/assets/img/prof_pic-800.webp"
      style="position: absolute; top: 0; right: 0;
            width:100px; height:95px;">

## Si Yudong

<span class="icon">&#xe60f;</span> `19121726080`&emsp;&emsp;
<span class="icon">&#xe7ca;</span> `1505632943@qq.com`&emsp;&emsp;
<span class="icon">&#xe600;</span> `https://github.com/yudongsi`

### &#xe80c; Education

<div class="entry-title">
    <h3>Tongji University</h3>
    <p>2019.09 - 2022.03</p>
</div>
<div class="entry-title">
    <h3>Nantong University</h3>
    <p>2014.09 - 2018.03</p>
</div>

### &#xe618; Work Experience

<div class="entry-title">
    <h3>AI Framework Engineer (Triton Compiler)@Intel.</h3>
    <p>2022.07 - 2025.07</p>
</div>

#### Triton Compiler XPU Backend Development

- **Feature Implementation**
  - Implemented global scratch memory addressing in backend launcher
  - Built SPIR-V → Level Zero → SYCL AOT compilation pipeline, facilitating low-level issue reproduction.
  - Created Intel specific Pass tritonintelgpu-rewrite-stack-ptr to optimize shared local memory (SLM) pointer handling.
  - Integrated \_\_spirv_RoundFToTF32INTEL SPIRV extension for enhanced FP32 precision in tl.dot DPAS lowering.
- **Performance Optimization**
  - Established Triton-XPU's first benchmark system covering Softmax/GEMM/FA key kernels, with extensible support for vendor libraries (XeTLA/CUTLASS/oneDNN).
  - Landing optimization passes like: Coalesce，AcceletateMatmul, RemoveLayoutConversion, Pipeline / Prefetch, Swizzling etc.
  - Optimized critical kernels (e.g. GEMM) and achieved >90% performance of Intel's XeTLA library
- **Bug Fixes**
  - Resolved 35+ High-priority compiler backend issues including: performance regressions, Layout propagation failures in IR passes, ut issues and so on.

#### PyTorch Ecosystem Optimization

- **CI/CD Innovation**
  - Designed AWS Xeon-based Jenkins pipeline for PyTorch Inductor CPU Performance:
    - Automated collection of 200+ model performance metrics
    - Auto-generated performance reports

<div class="entry-title">
    <h3>CUDA Test Engineer Intern@NVIDIA</h3>
    <p>2021.08 - 2021.11</p>
</div>

#### CUDA Orin Simulator Practice

- **Built CUDA safety & code coverage system for NVIDIA Orin t23x SOC**
  - Automated test platform using VDK virtual test suite
  - Debugged failed/timeout test cases
  - Automated test image version tracking

### &#xecfa; Skills

- **Compiler**: Triton,LLVM,MLIR,SPIRV
- **Heterogeneous Computing**: oneAPI,OpenCL,CUDA,SYCL,Level Zero,ROCm
- **AI Framework**: Pytorch,IPEX
- **AI Infrastructure**: Jenkins,GitHub Actions,Docker,Vtune Profiler
- **Languages**: C/C++, Python, Bash
