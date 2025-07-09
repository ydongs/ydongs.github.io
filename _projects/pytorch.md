---
layout: page
title: pytorch/pytorch
description: Tensors and Dynamic neural networks in Python with strong GPU acceleration
img: assets/img/pytorch-logo.png
importance: 2
category: work
toc:
  beginning: true

shortcuts:
  - name: Code
    icon: fa-brands fa-github
    link: https://github.com/pytorch/pytorch
---

# Introduction

We built a CI/CD pipeline on AWS Xeon instances using Jenkins to automatically collect, test, and publish PyTorch Inductor CPU performance data – accelerating Intel's optimization efforts for PyTorch.

# Details

Intel's blog: [Accelerated CPU Inference with PyTorch\* Inductor using torch.compile](https://www.intel.com/content/www/us/en/developer/articles/technical/accelerated-cpu-inference-with-pytorch-inductor.html)

Pytorch's readme: [cpu-performance-dashboard](https://github.com/pytorch/pytorch/blob/main/benchmarks/dynamo/README.md#cpu-performance-dashboard)

[Pytorch benchmarks](https://github.com/pytorch/pytorch/tree/main/benchmarks/dynamo) includes three main benchmark suites: [TorchBench](https://github.com/pytorch/benchmark), [Hugging Face](https://github.com/huggingface/transformers), and [TIMM](https://github.com/huggingface/pytorch-image-models), covered most popular models.

We build a [CI/CD infrastructure](https://github.com/chuanqi129/inductor-tools) infrastructure on [AWS](https://aws.amazon.com/intel) using [Jenkins](https://www.jenkins.io) pipelines to fully automate workflows, including bad commit bisection.

Data published on [Inductor CPU Performance Dashboard](https://github.com/pytorch/pytorch/issues/93531).

# Performance Dashboard for float32 precision -- Single-core Single-thread

## Executive Summary

<details>
<summary>see more</summary>
We evaluate different backends across three benchmark suites - torchbench, huggingface and timm. We run these experiments on Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz. Each experiment runs one iteration of forward pass. For accuracy, we check the numerical correctness of forward pass outputs by comparing with native pytorch. We measure speedup by normalizing against the performance of native pytorch. We report mean compilation latency numbers and peak memory footprint reduction ratio.

Caveats

1. Batch size has been reduced to workaround OOM errors. Work is in progress to reduce peak memory footprint.
2. Experiments do not cover dynamic shapes.
3. Experimental setup does not have optimizer.

</details>

To measure performance, compilation latency and memory footprint reduction, we remove the models that fail accuracy checks.

Passrate

```
+----------+------------+-------------+-------------+
| Compiler | torchbench | huggingface | timm_models |
+----------+------------+-------------+-------------+
| inductor | 91%, 49/54 | 100%, 44/44 | 89%, 54/61  |
+----------+------------+-------------+-------------+
```

Geometric mean speedup

```
+----------+------------+-------------+-------------+
| Compiler | torchbench | huggingface | timm_models |
+----------+------------+-------------+-------------+
| inductor |    1.04x   |    1.00x    |    1.07x    |
+----------+------------+-------------+-------------+
```

Mean compilation time (seconds)

```
+----------+------------+-------------+-------------+
| Compiler | torchbench | huggingface | timm_models |
+----------+------------+-------------+-------------+
| inductor |    17.21   |    19.50    |    22.93    |
+----------+------------+-------------+-------------+
```

Peak memory footprint compression ratio (higher is better)

```
+----------+------------+-------------+-------------+
| Compiler | torchbench | huggingface | timm_models |
+----------+------------+-------------+-------------+
| inductor |    1.23x   |     1.30x   |    1.05x    |
+----------+------------+-------------+-------------+
```

## torchbench suite with float32 precision

<details>
<summary>see more</summary>

Performance speedup

```
+-----------------------------------+----+----------+
|               name                | bs | inductor |
+-----------------------------------+----+----------+
|        shufflenet_v2_x1_0         | 1  |  1.4924  |
|           squeezenet1_1           | 1  |  1.2456  |
|   pytorch_CycleGAN_and_pix2pix    | 1  |  1.1829  |
|             resnet18              | 1  |  1.1615  |
|       functorch_dp_cifar10        | 1  |  1.1532  |
|           pytorch_unet            | 1  |  1.1414  |
| attention_is_all_you_need_pytorch | 1  |  1.1106  |
|            timm_vovnet            | 1  |  1.1089  |
|            Super_SloMo            | 1  |  1.0927  |
|               vgg16               | 1  |  1.0908  |
|              alexnet              | 1  |  1.0802  |
|                drq                | 1  |  1.0768  |
|            timm_regnet            | 1  |  1.0688  |
|        Background_Matting         | 1  |  1.0576  |
|          LearningToPaint          | 1  |  1.0514  |
|               dlrm                | 1  |  1.0468  |
|            densenet121            | 1  |  1.0435  |
|           timm_resnest            | 1  |  1.042   |
|               dcgan               | 1  |  1.0316  |
|          pytorch_stargan          | 16 |  1.0133  |
|              demucs               | 1  |  0.9988  |
|            tts_angular            | 1  |  0.9982  |
|      resnet50_quantized_qat       | 1  |  0.9973  |
|            hf_BigBird             | 1  |  0.9936  |
|    mobilenet_v2_quantized_qat     | 1  |  0.9924  |
|            timm_nfnet             | 1  |  0.9911  |
|      nvidia_deeprecommender       | 1  |  0.9847  |
|           mobilenet_v2            | 1  |  0.9292  |
|            mnasnet1_0             | 1  |  0.9092  |
|              yolov3               | 1  |  0.8915  |
|              hf_GPT2              | 1  |  0.8666  |
|             resnet50              | 1  |  0.8611  |
|           hf_Longformer           | 1  |  0.8127  |
|           BERT_pytorch            | 1  |  0.8125  |
|            hf_T5_large            | 1  |  0.8077  |
|             hf_Albert             | 1  |  0.8057  |
|   timm_vision_transformer_large   | 1  |  0.8052  |
|            hf_Reformer            | 1  |  0.7879  |
|        mobilenet_v3_large         | 1  |  0.7861  |
|           lennard_jones           | 1  |  0.7822  |
|          resnext50_32x4d          | 1  |  0.7543  |
|           hf_GPT2_large           | 1  |  0.7516  |
|               hf_T5               | 1  |  0.7441  |
|           hf_DistilBert           | 1  |  0.7326  |
|              hf_Bart              | 1  |  0.7097  |
|            hf_T5_base             | 1  |  0.6893  |
|              hf_Bert              | 1  |  0.6852  |
|      timm_vision_transformer      | 1  |  0.5967  |
|         timm_efficientnet         | 1  |  0.5114  |
|         soft_actor_critic         | 0  |   0.0    |
|        speech_transformer         | 0  |   0.0    |
|         timm_efficientdet         | 0  |   0.0    |
|           fastNLP_Bert            | 0  |   0.0    |
|             tacotron2             | 0  |   0.0    |
+-----------------------------------+----+----------+
```

Accuracy

```
+-----------------------------------+----+------------------+
|               name                | bs |     inductor     |
+-----------------------------------+----+------------------+
|            hf_T5_large            | 1  | pass_due_to_skip |
|           hf_GPT2_large           | 1  | pass_due_to_skip |
|   timm_vision_transformer_large   | 1  | pass_due_to_skip |
|               hf_T5               | 1  |       pass       |
|            hf_Reformer            | 1  |       pass       |
|            hf_T5_base             | 1  |       pass       |
|          LearningToPaint          | 1  |       pass       |
|            Super_SloMo            | 1  |       pass       |
|              alexnet              | 1  |       pass       |
| attention_is_all_you_need_pytorch | 1  |       pass       |
|               dcgan               | 1  |       pass       |
|              demucs               | 1  |       pass       |
|            densenet121            | 1  |       pass       |
|               dlrm                | 1  |       pass       |
|                drq                | 1  |       pass       |
|              yolov3               | 1  |       pass       |
|           mobilenet_v2            | 1  |       pass       |
|             hf_Albert             | 1  |       pass       |
|              hf_Bart              | 1  |       pass       |
|              hf_Bert              | 1  |       pass       |
|            hf_BigBird             | 1  |       pass       |
|           hf_DistilBert           | 1  |       pass       |
|              hf_GPT2              | 1  |       pass       |
|           hf_Longformer           | 1  |       pass       |
|       functorch_dp_cifar10        | 1  |       pass       |
|           lennard_jones           | 1  |       pass       |
|        Background_Matting         | 1  |       pass       |
|          resnext50_32x4d          | 1  |       pass       |
|           BERT_pytorch            | 1  |       pass       |
|      resnet50_quantized_qat       | 1  |       pass       |
|    mobilenet_v2_quantized_qat     | 1  |       pass       |
|        mobilenet_v3_large         | 1  |       pass       |
|      nvidia_deeprecommender       | 1  |       pass       |
|   pytorch_CycleGAN_and_pix2pix    | 1  |       pass       |
|          pytorch_stargan          | 16 |       pass       |
|           pytorch_unet            | 1  |       pass       |
|               vgg16               | 1  |       pass       |
|             resnet50              | 1  |       pass       |
|             resnet18              | 1  |       pass       |
|            mnasnet1_0             | 1  |       pass       |
|        shufflenet_v2_x1_0         | 1  |       pass       |
|           squeezenet1_1           | 1  |       pass       |
|         timm_efficientnet         | 1  |       pass       |
|            timm_nfnet             | 1  |       pass       |
|            timm_regnet            | 1  |       pass       |
|           timm_resnest            | 1  |       pass       |
|      timm_vision_transformer      | 1  |       pass       |
|            timm_vovnet            | 1  |       pass       |
|            tts_angular            | 1  |       pass       |
|         soft_actor_critic         | 0  |      0.0000      |
|           fastNLP_Bert            | 0  |      0.0000      |
|             tacotron2             | 0  |      0.0000      |
|         timm_efficientdet         | 0  |      0.0000      |
|        speech_transformer         | 0  |      0.0000      |
+-----------------------------------+----+------------------+
```

Compilation latency (sec)

```
+-----------------------------------+----+----------+
|               name                | bs | inductor |
+-----------------------------------+----+----------+
|            hf_T5_base             | 1  | 76.7783  |
|            hf_T5_large            | 1  | 60.5122  |
|           hf_GPT2_large           | 1  | 59.4694  |
|            densenet121            | 1  | 30.8545  |
|   timm_vision_transformer_large   | 1  | 28.9246  |
|            timm_nfnet             | 1  | 25.4024  |
|              yolov3               | 1  | 24.6705  |
|           hf_Longformer           | 1  | 24.4957  |
|            Super_SloMo            | 1  | 24.3262  |
|         timm_efficientnet         | 1  | 23.1212  |
|        Background_Matting         | 1  | 22.8918  |
|            hf_BigBird             | 1  | 22.3391  |
|           pytorch_unet            | 1  | 18.9934  |
|            timm_regnet            | 1  | 18.4639  |
|            hf_Reformer            | 1  | 18.0788  |
|        mobilenet_v3_large         | 1  | 17.9186  |
|              hf_Bart              | 1  | 17.7819  |
|               hf_T5               | 1  | 17.7063  |
|            timm_vovnet            | 1  | 15.5172  |
|          resnext50_32x4d          | 1  | 14.7342  |
|             resnet50              | 1  | 14.6658  |
|              hf_Bert              | 1  | 14.5967  |
|           mobilenet_v2            | 1  |  14.549  |
|              hf_GPT2              | 1  | 13.9911  |
|           BERT_pytorch            | 1  | 13.8472  |
|           hf_DistilBert           | 1  | 13.8324  |
|           timm_resnest            | 1  |  13.25   |
|            mnasnet1_0             | 1  | 13.1723  |
|        shufflenet_v2_x1_0         | 1  | 12.8739  |
|             hf_Albert             | 1  | 12.6541  |
|      timm_vision_transformer      | 1  | 12.2517  |
| attention_is_all_you_need_pytorch | 1  | 11.9372  |
|       functorch_dp_cifar10        | 1  | 11.8487  |
|          pytorch_stargan          | 16 | 11.5192  |
|          LearningToPaint          | 1  | 10.9144  |
|   pytorch_CycleGAN_and_pix2pix    | 1  | 10.1494  |
|           squeezenet1_1           | 1  |  9.7543  |
|               vgg16               | 1  |  8.5459  |
|             resnet18              | 1  |  8.449   |
|      nvidia_deeprecommender       | 1  |  8.3243  |
|                drq                | 1  |  8.1052  |
|               dlrm                | 1  |  7.9679  |
|              alexnet              | 1  |  7.3684  |
|            tts_angular            | 1  |  6.9832  |
|           lennard_jones           | 1  |  6.5252  |
|              demucs               | 1  |  1.4428  |
|               dcgan               | 1  |  0.2565  |
|    mobilenet_v2_quantized_qat     | 1  |  0.174   |
|      resnet50_quantized_qat       | 1  |  0.1546  |
|           fastNLP_Bert            | 0  |   nan    |
|         soft_actor_critic         | 0  |   nan    |
|        speech_transformer         | 0  |   nan    |
|             tacotron2             | 0  |   nan    |
|         timm_efficientdet         | 0  |   nan    |
+-----------------------------------+----+----------+
```

Peak Memory Compression Ratio

```
+-----------------------------------+----+----------+
|               name                | bs | inductor |
+-----------------------------------+----+----------+
|            hf_T5_base             | 1  |  4.3494  |
|           pytorch_unet            | 1  |  2.1317  |
|             hf_Albert             | 1  |  2.1001  |
|           hf_GPT2_large           | 1  |  1.9307  |
|            hf_BigBird             | 1  |  1.7971  |
|            Super_SloMo            | 1  |  1.7855  |
|            hf_T5_large            | 1  |  1.5672  |
|   pytorch_CycleGAN_and_pix2pix    | 1  |  1.4777  |
|               hf_T5               | 1  |  1.3382  |
|              hf_Bart              | 1  |  1.3275  |
|              hf_Bert              | 1  |  1.2964  |
|           hf_Longformer           | 1  |  1.2845  |
|               dlrm                | 1  |  1.2155  |
|          LearningToPaint          | 1  |  1.1866  |
|           hf_DistilBert           | 1  |  1.1789  |
|            timm_nfnet             | 1  |  1.1521  |
|            timm_regnet            | 1  |  1.1463  |
|              hf_GPT2              | 1  |  1.1158  |
|   timm_vision_transformer_large   | 1  |  1.1142  |
|                drq                | 1  |  1.0669  |
|            timm_vovnet            | 1  |  1.0587  |
|           mobilenet_v2            | 1  |  1.0514  |
|         timm_efficientnet         | 1  |  1.0509  |
|      timm_vision_transformer      | 1  |  1.0436  |
|              yolov3               | 1  |  1.0023  |
|              demucs               | 1  |  0.9985  |
|      nvidia_deeprecommender       | 1  |  0.9962  |
|      resnet50_quantized_qat       | 1  |  0.9955  |
|        Background_Matting         | 1  |  0.9953  |
|    mobilenet_v2_quantized_qat     | 1  |  0.9928  |
|            tts_angular            | 1  |  0.9921  |
|          pytorch_stargan          | 16 |  0.9883  |
|            mnasnet1_0             | 1  |  0.9844  |
|            densenet121            | 1  |  0.9831  |
| attention_is_all_you_need_pytorch | 1  |  0.983   |
|           lennard_jones           | 1  |  0.9818  |
|              alexnet              | 1  |  0.9803  |
|        mobilenet_v3_large         | 1  |  0.9771  |
|               vgg16               | 1  |  0.9758  |
|          resnext50_32x4d          | 1  |  0.975   |
|           squeezenet1_1           | 1  |  0.966   |
|               dcgan               | 1  |  0.9639  |
|        shufflenet_v2_x1_0         | 1  |  0.9582  |
|           timm_resnest            | 1  |  0.9582  |
|       functorch_dp_cifar10        | 1  |  0.9542  |
|             resnet50              | 1  |  0.9529  |
|            hf_Reformer            | 1  |  0.9437  |
|           BERT_pytorch            | 1  |  0.9242  |
|             resnet18              | 1  |  0.917   |
|           fastNLP_Bert            | 0  |   nan    |
|         soft_actor_critic         | 0  |   nan    |
|        speech_transformer         | 0  |   nan    |
|             tacotron2             | 0  |   nan    |
|         timm_efficientdet         | 0  |   nan    |
+-----------------------------------+----+----------+
```

</details>

## huggingface suite with float32 precision

<details>
<summary>see more</summary>

Performance speedup

```
+-----------------------------------------+----+----------+
|                  name                   | bs | inductor |
+-----------------------------------------+----+----------+
|     MobileBertForQuestionAnswering      | 1  |  1.0761  |
|            XLNetLMHeadModel             | 1  |  1.049   |
|       AlbertForQuestionAnswering        | 1  |  0.9522  |
|            AlbertForMaskedLM            | 1  |  0.9483  |
|          MobileBertForMaskedLM          | 1  |  0.9464  |
|                 BigBird                 | 1  |  0.9053  |
|     M2M100ForConditionalGeneration      | 1  |  0.9032  |
|             OPTForCausalLM              | 1  |  0.8764  |
|               GoogleFnet                | 1  |  0.8565  |
|            YituTechConvBert             | 1  |  0.8172  |
|         Speech2Text2ForCausalLM         | 1  |  0.8158  |
|    MegatronBertForQuestionAnswering     | 1  |  0.8047  |
|     PegasusForConditionalGeneration     | 1  |  0.8046  |
|       DebertaForQuestionAnswering       | 1  |  0.8035  |
|      MBartForConditionalGeneration      | 1  |  0.8029  |
|         MegatronBertForCausalLM         | 1  |  0.7995  |
|       RobertaForQuestionAnswering       | 1  |  0.7913  |
|          AllenaiLongformerBase          | 1  |  0.791   |
|            TrOCRForCausalLM             | 1  |  0.7903  |
|           PegasusForCausalLM            | 1  |  0.7878  |
|             XGLMForCausalLM             | 1  |  0.7832  |
|            MBartForCausalLM             | 1  |  0.7824  |
|           RobertaForCausalLM            | 1  |  0.7728  |
|     PLBartForConditionalGeneration      | 1  |  0.7699  |
|     DistilBertForQuestionAnswering      | 1  |  0.7654  |
|           DebertaForMaskedLM            | 1  |  0.7648  |
|        BertForQuestionAnswering         | 1  |  0.756   |
|             BertForMaskedLM             | 1  |  0.7524  |
|            PLBartForCausalLM            | 1  |  0.7459  |
|          DistilBertForMaskedLM          | 1  |  0.7431  |
|               DistillGPT2               | 1  |  0.726   |
|       MT5ForConditionalGeneration       | 1  |  0.7145  |
|      GPT2ForSequenceClassification      | 1  |  0.6909  |
|           LayoutLMForMaskedLM           | 1  |  0.6807  |
| BlenderbotSmallForConditionalGeneration | 1  |   0.68   |
|    LayoutLMForSequenceClassification    | 1  |  0.6773  |
|             BartForCausalLM             | 1  |  0.6707  |
|                CamemBert                | 1  |   0.67   |
|       BlenderbotSmallForCausalLM        | 1  |  0.6692  |
|      BartForConditionalGeneration       | 1  |  0.6425  |
|       T5ForConditionalGeneration        | 1  |  0.6322  |
|                 T5Small                 | 1  |  0.6311  |
|       ElectraForQuestionAnswering       | 1  |  0.5028  |
|           ElectraForCausalLM            | 1  |  0.4886  |
+-----------------------------------------+----+----------+
```

Accuracy

```
+-----------------------------------------+----+----------+
|                  name                   | bs | inductor |
+-----------------------------------------+----+----------+
|            AlbertForMaskedLM            | 1  |   pass   |
|       AlbertForQuestionAnswering        | 1  |   pass   |
|                CamemBert                | 1  |   pass   |
|          AllenaiLongformerBase          | 1  |   pass   |
|             BartForCausalLM             | 1  |   pass   |
|      BartForConditionalGeneration       | 1  |   pass   |
|             BertForMaskedLM             | 1  |   pass   |
|        BertForQuestionAnswering         | 1  |   pass   |
|                 BigBird                 | 1  |   pass   |
|       BlenderbotSmallForCausalLM        | 1  |   pass   |
| BlenderbotSmallForConditionalGeneration | 1  |   pass   |
|           DebertaForMaskedLM            | 1  |   pass   |
|           LayoutLMForMaskedLM           | 1  |   pass   |
|       DebertaForQuestionAnswering       | 1  |   pass   |
|          DistilBertForMaskedLM          | 1  |   pass   |
|     DistilBertForQuestionAnswering      | 1  |   pass   |
|               DistillGPT2               | 1  |   pass   |
|           ElectraForCausalLM            | 1  |   pass   |
|       ElectraForQuestionAnswering       | 1  |   pass   |
|      GPT2ForSequenceClassification      | 1  |   pass   |
|               GoogleFnet                | 1  |   pass   |
|    LayoutLMForSequenceClassification    | 1  |   pass   |
|     M2M100ForConditionalGeneration      | 1  |   pass   |
|            MBartForCausalLM             | 1  |   pass   |
|     PLBartForConditionalGeneration      | 1  |   pass   |
|      MBartForConditionalGeneration      | 1  |   pass   |
|       MT5ForConditionalGeneration       | 1  |   pass   |
|         MegatronBertForCausalLM         | 1  |   pass   |
|    MegatronBertForQuestionAnswering     | 1  |   pass   |
|          MobileBertForMaskedLM          | 1  |   pass   |
|     MobileBertForQuestionAnswering      | 1  |   pass   |
|             OPTForCausalLM              | 1  |   pass   |
|            PLBartForCausalLM            | 1  |   pass   |
|           PegasusForCausalLM            | 1  |   pass   |
|            XLNetLMHeadModel             | 1  |   pass   |
|     PegasusForConditionalGeneration     | 1  |   pass   |
|           RobertaForCausalLM            | 1  |   pass   |
|       RobertaForQuestionAnswering       | 1  |   pass   |
|         Speech2Text2ForCausalLM         | 1  |   pass   |
|       T5ForConditionalGeneration        | 1  |   pass   |
|                 T5Small                 | 1  |   pass   |
|            TrOCRForCausalLM             | 1  |   pass   |
|             XGLMForCausalLM             | 1  |   pass   |
|            YituTechConvBert             | 1  |   pass   |
+-----------------------------------------+----+----------+
```

Compilation latency (sec)

```
+-----------------------------------------+----+----------+
|                  name                   | bs | inductor |
+-----------------------------------------+----+----------+
|      BartForConditionalGeneration       | 1  | 49.6995  |
|            AlbertForMaskedLM            | 1  | 32.9211  |
|          AllenaiLongformerBase          | 1  |  31.457  |
|       AlbertForQuestionAnswering        | 1  | 30.8554  |
|            YituTechConvBert             | 1  | 25.3315  |
|             BartForCausalLM             | 1  | 24.3902  |
|           PegasusForCausalLM            | 1  |  23.914  |
|          MobileBertForMaskedLM          | 1  | 23.6263  |
|            XLNetLMHeadModel             | 1  |  23.513  |
|     PegasusForConditionalGeneration     | 1  | 23.4144  |
|     M2M100ForConditionalGeneration      | 1  | 23.2889  |
|     MobileBertForQuestionAnswering      | 1  | 23.2762  |
|                 BigBird                 | 1  | 22.9858  |
|      MBartForConditionalGeneration      | 1  | 22.3898  |
|           DebertaForMaskedLM            | 1  |  22.357  |
|       T5ForConditionalGeneration        | 1  |  22.172  |
|       MT5ForConditionalGeneration       | 1  | 22.1254  |
|             XGLMForCausalLM             | 1  | 22.0928  |
|                 T5Small                 | 1  | 21.8409  |
|       DebertaForQuestionAnswering       | 1  | 21.3565  |
|         MegatronBertForCausalLM         | 1  | 20.1642  |
|             BertForMaskedLM             | 1  | 20.0772  |
|    MegatronBertForQuestionAnswering     | 1  | 19.6571  |
|      GPT2ForSequenceClassification      | 1  | 19.1087  |
|               GoogleFnet                | 1  | 18.4626  |
| BlenderbotSmallForConditionalGeneration | 1  | 15.2419  |
|                CamemBert                | 1  | 15.1637  |
|     PLBartForConditionalGeneration      | 1  | 14.9998  |
|           LayoutLMForMaskedLM           | 1  | 14.9718  |
|            MBartForCausalLM             | 1  |  14.451  |
|    LayoutLMForSequenceClassification    | 1  | 14.2165  |
|        BertForQuestionAnswering         | 1  | 13.9057  |
|             OPTForCausalLM              | 1  | 13.7452  |
|               DistillGPT2               | 1  |  13.034  |
|            TrOCRForCausalLM             | 1  | 12.9223  |
|           ElectraForCausalLM            | 1  | 12.9222  |
|           RobertaForCausalLM            | 1  | 12.9136  |
|       RobertaForQuestionAnswering       | 1  | 12.6522  |
|       ElectraForQuestionAnswering       | 1  | 12.3606  |
|       BlenderbotSmallForCausalLM        | 1  | 11.2526  |
|         Speech2Text2ForCausalLM         | 1  | 11.1537  |
|            PLBartForCausalLM            | 1  | 10.7109  |
|          DistilBertForMaskedLM          | 1  | 10.5215  |
|     DistilBertForQuestionAnswering      | 1  | 10.2008  |
+-----------------------------------------+----+----------+
```

Peak Memory Compression Ratio

```
+-----------------------------------------+----+----------+
|                  name                   | bs | inductor |
+-----------------------------------------+----+----------+
|            AlbertForMaskedLM            | 1  |  2.7784  |
|       AlbertForQuestionAnswering        | 1  |  2.726   |
|      BartForConditionalGeneration       | 1  |  2.1303  |
|       T5ForConditionalGeneration        | 1  |  1.8725  |
|                 T5Small                 | 1  |  1.8719  |
|                 BigBird                 | 1  |  1.8559  |
|      GPT2ForSequenceClassification      | 1  |  1.8462  |
|             BartForCausalLM             | 1  |  1.515   |
|          AllenaiLongformerBase          | 1  |  1.486   |
|            XLNetLMHeadModel             | 1  |  1.3611  |
|       DebertaForQuestionAnswering       | 1  |  1.3577  |
|                CamemBert                | 1  |  1.3398  |
|           LayoutLMForMaskedLM           | 1  |  1.3339  |
|            YituTechConvBert             | 1  |  1.3279  |
|           DebertaForMaskedLM            | 1  |  1.3117  |
|               GoogleFnet                | 1  |  1.2932  |
|    LayoutLMForSequenceClassification    | 1  |  1.2861  |
|           ElectraForCausalLM            | 1  |  1.2708  |
|               DistillGPT2               | 1  |  1.1948  |
|       ElectraForQuestionAnswering       | 1  |  1.1867  |
|     PegasusForConditionalGeneration     | 1  |  1.1209  |
|      MBartForConditionalGeneration      | 1  |  1.0693  |
|       MT5ForConditionalGeneration       | 1  |  1.0645  |
|             BertForMaskedLM             | 1  |  1.0616  |
|     M2M100ForConditionalGeneration      | 1  |  1.0565  |
|         MegatronBertForCausalLM         | 1  |  1.056   |
|           RobertaForCausalLM            | 1  |  1.0498  |
|            TrOCRForCausalLM             | 1  |  1.0498  |
|    MegatronBertForQuestionAnswering     | 1  |  1.0494  |
|       BlenderbotSmallForCausalLM        | 1  |  1.0494  |
|             OPTForCausalLM              | 1  |  1.0467  |
|        BertForQuestionAnswering         | 1  |  1.038   |
|          DistilBertForMaskedLM          | 1  |  1.0359  |
|       RobertaForQuestionAnswering       | 1  |  1.0347  |
|             XGLMForCausalLM             | 1  |  1.0325  |
|     PLBartForConditionalGeneration      | 1  |  1.0286  |
| BlenderbotSmallForConditionalGeneration | 1  |  1.0284  |
|            MBartForCausalLM             | 1  |  1.026   |
|            PLBartForCausalLM            | 1  |  1.0254  |
|     DistilBertForQuestionAnswering      | 1  |  1.017   |
|           PegasusForCausalLM            | 1  |  1.0034  |
|          MobileBertForMaskedLM          | 1  |  0.9964  |
|         Speech2Text2ForCausalLM         | 1  |  0.9816  |
|     MobileBertForQuestionAnswering      | 1  |  0.9741  |
+-----------------------------------------+----+----------+
```

</details>

## timm_models suite with float32 precision

<details>
<summary>see more</summary>

Performance speedup

```
+---------------------------------+----+----------+
|              name               | bs | inductor |
+---------------------------------+----+----------+
|          pnasnet5large          | 1  |  1.5744  |
|           regnety_002           | 1  |  1.3378  |
|        ese_vovnet19b_dw         | 1  |  1.3321  |
|          inception_v3           | 1  |  1.332   |
|       gluon_inception_v3        | 1  |  1.3149  |
|           mnasnet_100           | 1  |  1.3143  |
|        adv_inception_v3         | 1  |  1.3085  |
|            lcnet_050            | 1  |  1.2876  |
|          spnasnet_100           | 1  |  1.2836  |
|           fbnetc_100            | 1  |  1.2526  |
|         mobilenetv2_100         | 1  |  1.2212  |
|            fbnetv3_b            | 1  |   1.16   |
|      mobilenetv3_large_100      | 1  |  1.1506  |
|            gernet_l             | 1  |  1.0846  |
|             dpn107              | 1  |  1.0695  |
|        gluon_xception65         | 1  |  1.0594  |
|          cspdarknet53           | 1  |  1.0591  |
|            repvgg_a2            | 1  |  1.0524  |
|            hrnet_w18            | 1  |  1.0432  |
|          ghostnet_100           | 1  |  1.0058  |
|            nfnet_l0             | 1  |  1.0031  |
|           resnest101e           | 1  |  0.9801  |
|         crossvit_9_240          | 1  |  0.9629  |
|        twins_pcpvt_base         | 1  |  0.9453  |
|           selecsls42b           | 1  |  0.9185  |
|           dm_nfnet_f0           | 1  |  0.9119  |
|         visformer_small         | 1  |  0.9093  |
|        res2net101_26w_4s        | 1  |  0.9033  |
|      xcit_large_24_p8_224       | 1  |  0.9028  |
|          convnext_base          | 1  |  0.8551  |
|        res2net50_14w_8s         | 1  |  0.8483  |
|      beit_base_patch16_224      | 1  |  0.8427  |
|          gmixer_24_224          | 1  |  0.8267  |
| deit_base_distilled_patch16_224 | 1  |   0.79   |
|           res2next50            | 1  |  0.7897  |
|  swin_base_patch4_window7_224   | 1  |  0.7765  |
|           convit_base           | 1  |  0.7703  |
|          cait_m36_384           | 1  |  0.7632  |
|      vit_base_patch16_224       | 1  |  0.761   |
|           volo_d1_224           | 1  |  0.7515  |
|         poolformer_m36          | 1  |  0.736   |
|            mixnet_l             | 1  |  0.729   |
|             dla102              | 1  |  0.7196  |
|           tf_mixnet_l           | 1  |  0.7186  |
|          mixer_b16_224          | 1  |  0.7082  |
|          resmlp_12_224          | 1  |  0.666   |
|            pit_b_224            | 1  |  0.6644  |
|           rexnet_100            | 1  |  0.6578  |
|        tnt_s_patch16_224        | 1  |  0.6416  |
|          gmlp_s16_224           | 1  |  0.6327  |
|          jx_nest_base           | 1  |  0.6187  |
|           mobilevit_s           | 1  |  0.5702  |
|            tinynet_a            | 1  |  0.5466  |
|         coat_lite_mini          | 1  |  0.5378  |
|       tf_efficientnet_b0        | 1  |  0.5215  |
|     swsl_resnext101_32x16d      | 1  |  0.0669  |
|        sebotnet33ts_256         | 0  |   0.0    |
|       eca_botnext26ts_256       | 0  |   0.0    |
|        eca_halonext26ts         | 0  |   0.0    |
|          botnet26t_256          | 0  |   0.0    |
|        convmixer_768_32         | 0  |   0.0    |
+---------------------------------+----+----------+
```

Accuracy

```
+---------------------------------+----+---------------+
|              name               | bs |   inductor    |
+---------------------------------+----+---------------+
|        adv_inception_v3         | 1  |     pass      |
|             dpn107              | 1  |     pass      |
|      beit_base_patch16_224      | 1  |     pass      |
|          mixer_b16_224          | 1  |     pass      |
|        ese_vovnet19b_dw         | 1  |     pass      |
|         coat_lite_mini          | 1  |     pass      |
|           convit_base           | 1  |     pass      |
|          convnext_base          | 1  |     pass      |
|         crossvit_9_240          | 1  |     pass      |
|          cspdarknet53           | 1  |     pass      |
| deit_base_distilled_patch16_224 | 1  |     pass      |
|             dla102              | 1  |     pass      |
|           dm_nfnet_f0           | 1  |     pass      |
|            lcnet_050            | 1  |     pass      |
|           volo_d1_224           | 1  |     pass      |
|      xcit_large_24_p8_224       | 1  |     pass      |
|           fbnetc_100            | 1  |     pass      |
|        gluon_xception65         | 1  |     pass      |
|          jx_nest_base           | 1  |     pass      |
|          inception_v3           | 1  |     pass      |
|            hrnet_w18            | 1  |     pass      |
|          gmlp_s16_224           | 1  |     pass      |
|          gmixer_24_224          | 1  |     pass      |
|       gluon_inception_v3        | 1  |     pass      |
|            gernet_l             | 1  |     pass      |
|            fbnetv3_b            | 1  |     pass      |
|           mnasnet_100           | 1  |     pass      |
|            mixnet_l             | 1  |     pass      |
|      vit_base_patch16_224       | 1  |     pass      |
|           res2next50            | 1  |     pass      |
|         mobilenetv2_100         | 1  |     pass      |
|      mobilenetv3_large_100      | 1  |     pass      |
|           mobilevit_s           | 1  |     pass      |
|            nfnet_l0             | 1  |     pass      |
|            pit_b_224            | 1  |     pass      |
|          pnasnet5large          | 1  |     pass      |
|         poolformer_m36          | 1  |     pass      |
|           regnety_002           | 1  |     pass      |
|            repvgg_a2            | 1  |     pass      |
|         visformer_small         | 1  |     pass      |
|        res2net50_14w_8s         | 1  |     pass      |
|        res2net101_26w_4s        | 1  |     pass      |
|          resmlp_12_224          | 1  |     pass      |
|       tf_efficientnet_b0        | 1  |     pass      |
|        twins_pcpvt_base         | 1  |     pass      |
|        tnt_s_patch16_224        | 1  |     pass      |
|           resnest101e           | 1  |     pass      |
|           tf_mixnet_l           | 1  |     pass      |
|            tinynet_a            | 1  |     pass      |
|     swsl_resnext101_32x16d      | 1  |     pass      |
|  swin_base_patch4_window7_224   | 1  |     pass      |
|          spnasnet_100           | 1  |     pass      |
|           selecsls42b           | 1  |     pass      |
|           rexnet_100            | 1  |     pass      |
|        eca_halonext26ts         | 1  |  fail_to_run  |
|        convmixer_768_32         | 1  |  fail_to_run  |
|        sebotnet33ts_256         | 1  |  fail_to_run  |
|          botnet26t_256          | 1  |  fail_to_run  |
|       eca_botnext26ts_256       | 1  |  fail_to_run  |
|          ghostnet_100           | 1  | fail_accuracy |
|          cait_m36_384           | 1  | fail_accuracy |
+---------------------------------+----+---------------+
```

Compilation latency (sec)

```
+---------------------------------+----+----------+
|              name               | bs | inductor |
+---------------------------------+----+----------+
|     swsl_resnext101_32x16d      | 1  | 85.6242  |
|          pnasnet5large          | 1  | 43.6874  |
|            fbnetv3_b            | 1  | 34.4074  |
|          cait_m36_384           | 1  | 33.7081  |
|           tf_mixnet_l           | 1  | 33.6983  |
|        twins_pcpvt_base         | 1  | 31.5227  |
|            hrnet_w18            | 1  | 31.1337  |
|           rexnet_100            | 1  | 30.7918  |
|  swin_base_patch4_window7_224   | 1  | 30.3129  |
|           mobilevit_s           | 1  | 30.2527  |
|            mixnet_l             | 1  |  29.901  |
|      xcit_large_24_p8_224       | 1  | 28.4048  |
|             dpn107              | 1  | 26.7877  |
|        adv_inception_v3         | 1  | 26.4651  |
|        res2net50_14w_8s         | 1  | 26.3563  |
|           dm_nfnet_f0           | 1  | 26.1631  |
|          ghostnet_100           | 1  | 25.6518  |
|         poolformer_m36          | 1  | 24.9019  |
|       tf_efficientnet_b0        | 1  | 24.8772  |
|           resnest101e           | 1  | 24.5232  |
|         visformer_small         | 1  | 24.3597  |
|            tinynet_a            | 1  | 24.2595  |
|        res2net101_26w_4s        | 1  | 23.5626  |
|          jx_nest_base           | 1  | 23.1414  |
|         coat_lite_mini          | 1  | 22.4646  |
|            nfnet_l0             | 1  | 22.4105  |
|      mobilenetv3_large_100      | 1  | 22.3182  |
|             dla102              | 1  | 21.2783  |
|           volo_d1_224           | 1  | 21.2219  |
|         crossvit_9_240          | 1  | 21.2091  |
|        tnt_s_patch16_224        | 1  | 21.0931  |
|          convnext_base          | 1  | 19.4932  |
|          cspdarknet53           | 1  | 19.3065  |
|           fbnetc_100            | 1  | 18.9707  |
|       gluon_inception_v3        | 1  | 18.6493  |
|           res2next50            | 1  | 18.5211  |
|            pit_b_224            | 1  | 18.1465  |
|          spnasnet_100           | 1  | 18.1066  |
|          inception_v3           | 1  | 17.9952  |
|          gmlp_s16_224           | 1  | 17.9558  |
|      beit_base_patch16_224      | 1  |  17.513  |
|           regnety_002           | 1  | 17.0633  |
|        gluon_xception65         | 1  | 16.8521  |
|           mnasnet_100           | 1  | 16.5367  |
|         mobilenetv2_100         | 1  | 16.3189  |
|           convit_base           | 1  | 16.2035  |
|            gernet_l             | 1  | 14.7975  |
|          gmixer_24_224          | 1  |  14.685  |
|           selecsls42b           | 1  | 14.2609  |
|        ese_vovnet19b_dw         | 1  | 13.4973  |
|          mixer_b16_224          | 1  |  13.462  |
|      vit_base_patch16_224       | 1  | 13.3318  |
| deit_base_distilled_patch16_224 | 1  | 13.0111  |
|            repvgg_a2            | 1  | 12.9665  |
|            lcnet_050            | 1  | 12.8088  |
|          resmlp_12_224          | 1  | 10.4233  |
|          botnet26t_256          | 0  |   nan    |
|        convmixer_768_32         | 0  |   nan    |
|       eca_botnext26ts_256       | 0  |   nan    |
|        eca_halonext26ts         | 0  |   nan    |
|        sebotnet33ts_256         | 0  |   nan    |
+---------------------------------+----+----------+
```

Peak Memory Compression Ratio

```
+---------------------------------+----+----------+
|              name               | bs | inductor |
+---------------------------------+----+----------+
|      xcit_large_24_p8_224       | 1  |  2.2814  |
|          cait_m36_384           | 1  |  2.1693  |
|           dm_nfnet_f0           | 1  |  1.2079  |
|           mobilevit_s           | 1  |  1.2047  |
|          jx_nest_base           | 1  |  1.1988  |
|            nfnet_l0             | 1  |  1.1854  |
|             dpn107              | 1  |  1.1766  |
|        gluon_xception65         | 1  |  1.1551  |
|          convnext_base          | 1  |  1.155   |
|           convit_base           | 1  |  1.1399  |
|          cspdarknet53           | 1  |  1.1261  |
|         poolformer_m36          | 1  |  1.1142  |
|  swin_base_patch4_window7_224   | 1  |  1.106   |
|            pit_b_224            | 1  |  1.1013  |
|      beit_base_patch16_224      | 1  |  1.0972  |
|        twins_pcpvt_base         | 1  |  1.0907  |
|        tnt_s_patch16_224        | 1  |  1.0873  |
|           volo_d1_224           | 1  |  1.087   |
| deit_base_distilled_patch16_224 | 1  |  1.082   |
|      vit_base_patch16_224       | 1  |  1.0789  |
|          mixer_b16_224          | 1  |  1.0786  |
|         mobilenetv2_100         | 1  |  1.0611  |
|        ese_vovnet19b_dw         | 1  |  1.0596  |
|       tf_efficientnet_b0        | 1  |  1.057   |
|          resmlp_12_224          | 1  |  1.0457  |
|         coat_lite_mini          | 1  |  1.0422  |
|           rexnet_100            | 1  |  1.0408  |
|            mixnet_l             | 1  |  1.0377  |
|            gernet_l             | 1  |  1.0281  |
|            fbnetv3_b            | 1  |  1.0236  |
|            tinynet_a            | 1  |  1.0189  |
|           fbnetc_100            | 1  |  1.0121  |
|          spnasnet_100           | 1  |  1.0098  |
|           mnasnet_100           | 1  |  1.0097  |
|            repvgg_a2            | 1  |  1.0028  |
|         crossvit_9_240          | 1  |  0.9982  |
|           resnest101e           | 1  |  0.9928  |
|      mobilenetv3_large_100      | 1  |  0.9677  |
|            lcnet_050            | 1  |  0.9643  |
|           regnety_002           | 1  |  0.9604  |
|         visformer_small         | 1  |  0.9598  |
|          inception_v3           | 1  |  0.9554  |
|           res2next50            | 1  |  0.9538  |
|             dla102              | 1  |  0.9508  |
|        adv_inception_v3         | 1  |  0.9508  |
|            hrnet_w18            | 1  |  0.9504  |
|          ghostnet_100           | 1  |  0.9456  |
|        res2net50_14w_8s         | 1  |  0.9426  |
|       gluon_inception_v3        | 1  |  0.9393  |
|          gmixer_24_224          | 1  |  0.9237  |
|        res2net101_26w_4s        | 1  |  0.9135  |
|          gmlp_s16_224           | 1  |  0.8851  |
|     swsl_resnext101_32x16d      | 1  |  0.8458  |
|           tf_mixnet_l           | 1  |  0.8379  |
|           selecsls42b           | 1  |  0.8177  |
|          pnasnet5large          | 1  |  0.7496  |
|          botnet26t_256          | 0  |   nan    |
|        convmixer_768_32         | 0  |   nan    |
|       eca_botnext26ts_256       | 0  |   nan    |
|        eca_halonext26ts         | 0  |   nan    |
|        sebotnet33ts_256         | 0  |   nan    |
+---------------------------------+----+----------+
```

</details>
