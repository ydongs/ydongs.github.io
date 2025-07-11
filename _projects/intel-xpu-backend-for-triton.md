---
layout: page
title: intel/intel-xpu-backend-for-triton
description: OpenAI Triton backend for Intel® GPUs
img: assets/img/intel-logo.png
importance: 1
category: work
toc:
  beginning: true

shortcuts:
  - name: Code
    icon: fa-brands fa-github
    link: https://github.com/intel/intel-xpu-backend-for-triton
  - name: Contributions
    icon: fa-brands fa-github
    link: https://github.com/intel/intel-xpu-backend-for-triton/pulls?q=is%3Apr+is%3Amerged+author%3Ayudongsi

_styles: >
  div.triton-xpu-prs ul {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    list-style-position: inside;
    padding-left: 10px;
  }

  div.triton-xpu-prs ul > li {
    overflow-x: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
    margin-bottom: 0;
  }
---

# Introduction

This is the collection of my open source contributions to [intel-xpu-backend-for-triton](https://github.com/intel/intel-xpu-backend-for-triton).

_Intel® XPU Backend for Triton_, a new Triton backend for Intel GPUs. Intel® XPU Backend for Triton* is a out of tree backend module for Triton used to provide best-in-class performance and productivity on any Intel GPUs for PyTorch and standalone usage.*

# Contributions

## Summary

- [Contributor ranking](https://github.com/intel/intel-xpu-backend-for-triton/graphs/contributors).
- [Merged PRs](https://github.com/intel/intel-xpu-backend-for-triton/pulls?q=is%3Apr+is%3Amerged+author%3Ayudongsi).
- [Conversations involved](https://github.com/intel/intel-xpu-backend-for-triton/pulls?q=involves%3Ayudongsi).

## PRs

- **[#4484](https://github.com/intel/intel-xpu-backend-for-triton/pull/4484)** Use well tuned kernel options for flex attention
- **[#4271](https://github.com/intel/intel-xpu-backend-for-triton/pull/4271)** Support global scratch in launcher
- **[#4448](https://github.com/intel/intel-xpu-backend-for-triton/pull/4448)** Add softmax onednn impl in `benchmarks/triton_kernels_benchmark/fused_softmax.py`
- **[#4146](https://github.com/intel/intel-xpu-backend-for-triton/pull/4146)** Handle op with multi results case in changeAndPropagateLayout
- **[#3937](https://github.com/intel/intel-xpu-backend-for-triton/pull/3937)** Add `dot3d[8-2-64-64-64-32-32-float32-float32]` to `skiplists`
- **[#3875](https://github.com/intel/intel-xpu-backend-for-triton/pull/3875)** Make sure install `setuptools>=78.1.0` in `setup-triton`
- **[#3803](https://github.com/intel/intel-xpu-backend-for-triton/pull/3803)** add f32 rtne to tf32 in DPAS
- **[#3795](https://github.com/intel/intel-xpu-backend-for-triton/pull/3795)** Reland "Check the non 4-bytes aligned base/offsetX/width on block pointer (#3712)"
- **[#3712](https://github.com/intel/intel-xpu-backend-for-triton/pull/3712)** Check the non 4-bytes aligned base/offsetX/width on block pointer #3712"
- **[#3705](https://github.com/intel/intel-xpu-backend-for-triton/pull/3705)** [GEMM] Add the tensor of pointer benchmark
- **[#3644](https://github.com/intel/intel-xpu-backend-for-triton/pull/3644)** Fix `dpas_to_block_layout_convert.mlir`
- **[#3497](https://github.com/intel/intel-xpu-backend-for-triton/pull/3497)** Add `rewrite_stack_ptr` post process pass
- **[#3571](https://github.com/intel/intel-xpu-backend-for-triton/pull/3571)** Fix `test_reduce_layouts` for `LinearLayout`
- **[#3135](https://github.com/intel/intel-xpu-backend-for-triton/pull/3135)** Fix AOT compilation failed in Test with pip workflow
- **[#3108](https://github.com/intel/intel-xpu-backend-for-triton/pull/3108)** Add Flash Attention backward to `benchmarks/triton_kernels_benchmark`
- **[#2953](https://github.com/intel/intel-xpu-backend-for-triton/pull/2953)** Port and run tests in python/test/unit/tools
- **[#3010](https://github.com/intel/intel-xpu-backend-for-triton/pull/3010)** Fix `test_gather`
- **[#2839](https://github.com/intel/intel-xpu-backend-for-triton/pull/2839)** Improve performance of shape 1024x1024x1024 out of box
- **[#2646](https://github.com/intel/intel-xpu-backend-for-triton/pull/2646)** Improve GEMM performance of shape 4096x8x128x16384
- **[#2601](https://github.com/intel/intel-xpu-backend-for-triton/pull/2601)** Improve GEMM performance of shape 4096x8x128x16384
- **[#2520](https://github.com/intel/intel-xpu-backend-for-triton/pull/2520)** [XeTLA] Add xetla splitk gemm
- **[#2438](https://github.com/intel/intel-xpu-backend-for-triton/pull/2438)** [Benchmark] Run xetla streamk gemm in benchmark
- **[#2367](https://github.com/intel/intel-xpu-backend-for-triton/pull/2367)** Add `XeTLA` FA backward implementation to benchmark
- **[#2357](https://github.com/intel/intel-xpu-backend-for-triton/pull/2357)** Add causal variant in fa benchmark
- **[#2309](https://github.com/intel/intel-xpu-backend-for-triton/pull/2309)** [Benchmarks] Add more variants in XeTLA FA implementation
- **[#2060](https://github.com/intel/intel-xpu-backend-for-triton/pull/2060)** Add Triton benchmark support in compare script
- **[#2157](https://github.com/intel/intel-xpu-backend-for-triton/pull/2157)** Add attention adv path benchmark
- **[#1877](https://github.com/intel/intel-xpu-backend-for-triton/pull/1877)** Update XeTLA's attn implementation of Triton benchmark
- **[#1799](https://github.com/intel/intel-xpu-backend-for-triton/pull/1799)** Eliminate `XeTLA` GEMM performance gap
- **[#1714](https://github.com/intel/intel-xpu-backend-for-triton/pull/1714)** Add streamk xetla kernel
- **[#1741](https://github.com/intel/intel-xpu-backend-for-triton/pull/1741)** Remove cache triton in triton benchmark
- **[#1730](https://github.com/intel/intel-xpu-backend-for-triton/pull/1730)** [Benchmark] Fix xetla batch gemm cases
- **[#1707](https://github.com/intel/intel-xpu-backend-for-triton/pull/1707)** Add debug code for capture failure details
- **[#1695](https://github.com/intel/intel-xpu-backend-for-triton/pull/1695)** [BUG] Warkaround attr `allocation.offset` assertion failure
- **[#1597](https://github.com/intel/intel-xpu-backend-for-triton/pull/1597)** Update GEMM XeTLA kernel of triton benchmarks
- **[#1539](https://github.com/intel/intel-xpu-backend-for-triton/pull/1539)** Integrate flash attention XeTLA kernel into triton repo
- **[#1383](https://github.com/intel/intel-xpu-backend-for-triton/pull/1383)** Add triton bench deps step
- **[#1092](https://github.com/intel/intel-xpu-backend-for-triton/pull/1092)** [Performance] Clean and refine softmax and gemm benchmarks
- **[#877](https://github.com/intel/intel-xpu-backend-for-triton/pull/877)** [Performance] xetla kernels benchmark integration
- **[#977](https://github.com/intel/intel-xpu-backend-for-triton/pull/977)** enable block pointer gemm tutorial with new passes
- **[#845](https://github.com/intel/intel-xpu-backend-for-triton/pull/845)** [UT] use cpu result as reference for `fdiv`
- **[#614](https://github.com/intel/intel-xpu-backend-for-triton/pull/614)** Enable test_attention_fwd_bwd
- **[#308](https://github.com/intel/intel-xpu-backend-for-triton/pull/308)** minimize token permissions in workflows
- **[#246](https://github.com/intel/intel-xpu-backend-for-triton/pull/246)** [UT] Port and run operator tests
- **[#143](https://github.com/intel/intel-xpu-backend-for-triton/pull/143)** [ut] some operators and language cases
- **[#133](https://github.com/intel/intel-xpu-backend-for-triton/pull/133)** [CI] Update ut scope and pass rate calculation
- **[#136](https://github.com/intel/intel-xpu-backend-for-triton/pull/136)** [CI]Add dockerfiles
- **[#129](https://github.com/intel/intel-xpu-backend-for-triton/pull/129)** [CI] Refine CI workflows
- **[#127](https://github.com/intel/intel-xpu-backend-for-triton/pull/127)** [CI] Migrate action runners to dedicated one
- **[#124](https://github.com/intel/intel-xpu-backend-for-triton/pull/124)** [CI] add nightly failure notify support
- **[#120](https://github.com/intel/intel-xpu-backend-for-triton/pull/120)** update ZE_AFFINITY_MASK
- **[#112](https://github.com/intel/intel-xpu-backend-for-triton/pull/112)** upd triton hash
- **[#109](https://github.com/intel/intel-xpu-backend-for-triton/pull/109)** update usage on env_triton.sh
- **[#97](https://github.com/intel/intel-xpu-backend-for-triton/pull/97)** env prepare explicitly in workflows
- **[#81](https://github.com/intel/intel-xpu-backend-for-triton/pull/81)** add e2e perf test to nightly
- **[#60](https://github.com/intel/intel-xpu-backend-for-triton/pull/60)** fix wrong script path
- **[#59](https://github.com/intel/intel-xpu-backend-for-triton/pull/59)** reduce e2e test iterations
- **[#51](https://github.com/intel/intel-xpu-backend-for-triton/pull/51)** Add Inductor e2e workflow for triton xpu backend
