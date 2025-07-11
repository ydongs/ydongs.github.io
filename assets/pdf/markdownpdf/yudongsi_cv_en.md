<img src="https://ydongs.github.io/assets/img/prof_pic-800.webp"
      style="position: absolute; top: 0; right: 0;
            width:100px; height:95px;">

## Si Yudong

<span class="icon">&#xe60f;</span> `19121726080`&emsp;&emsp;
<span class="icon">&#xe7ca;</span> `1505632943@qq.com`&emsp;&emsp;
<span class="icon">&#xe600;</span> `https://github.com/ydongs`

### &#xe80c; Education

<div class="entry-title">
    <h3>Tongji University</h3>
    <p>2019.09 - 2022.03</p>
</div>
<div class="entry-title">
    <h3>Nantong University</h3>
    <p>2014.09 - 2018.06</p>
</div>

### &#xe618; Work Experience

<div class="entry-title">
    <h3>AI Framework Engineer (Triton Compiler)@Intel.</h3>
    <p>2022.07 - 2025.07</p>
</div>

#### Triton Compiler XPU Backend Development

- **Feature Implementation**
  - Developed GLM support, AOT compilation, TF32 SPIRV extensions
  - Built post-processing optimization pipeline (Postprocess Pass)
- **Performance Optimization**
  - Established first Triton-XPU Benchmark system (Softmax/GEMM/FA), supporting extendable handwritten libraries (XeTLA, CUTLASS, oneDNN)
  - Achieved >90% performance of Intel's XeTLA library in key kernels
- **Bug Fixes**
  - Resolved 35+ High-priority compiler backend issues

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

### &#xe635; Projects

<div class="entry-title">
    <h3>Driver Fatigue Detection System | Computer Vision | 2021.07 | Academic Project </h3>
</div>

Multi-feature fusion model (eye/mouth pose + head Euler angles)

- Designed joint detection: blink count (EAR<0.25) + yawn detection (MAR>0.8)
- Deployed lightweight inference pipeline on embedded board
- Developed QT monitoring UI with visual fatigue alerts

<div class="entry-title">
    <h3>Automotive Ambient Lighting Control | Embedded Systems | 2020.02 | Academic Project </h3>
</div>

Vehicle-grade programmable ambient light control system (touchscreen to LIN bus)

- Designed touchscreen UART command parser with custom instructions
- Implemented LIN bus unconditional frame protocol
- Created capacitive touch HMI interface

### &#xecfa; Skills

- **Languages**: C/C++, Python, Bash
- **Fundamentals**: Data Structures & Algorithms, OS, Computer Architecture
- **AI Infrastructure**: PyTorch,LLVM,MLIR,oneAPI,OpenCL,CUDA,SYCL,Jenkins,GitHub Actions,Docker
