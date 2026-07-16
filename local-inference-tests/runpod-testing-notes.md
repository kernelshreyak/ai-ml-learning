# RunPod Testing Devlog

---

# 2026-07-15 — Initial Exploration

## Overview

Spent the day exploring RunPod to become familiar with its interface, workflow, and storage options.

## Findings

| Topic | Details |
|---|---|
| **RunPod Flash** | Explored the platform and its available features. |
| **Interface** | Became familiar with the general workflow and user interface. |
| **Volume performance** | Network volumes are **significantly slower** than attached volumes. |
| **PyTorch experiments** | Successfully ran PyTorch experiments through Jupyter Notebook on a pod. |

---

# 2026-07-16 — Serverless Inference & GPU Benchmarks

## Serverless Setup

### Completed

- [x] API key configured.

---

## Fast Chat Model Testing (16–24 GB VRAM)

| Command | Result | Notes |
|---|---|---|
| `vllm serve "openchat/openchat-3.5-0106"` | ✅ Works | Successfully served; compatible with ChatBox and the OpenAI SDK. |
| `vllm serve "QuantTrio/Qwen3.5-27B-AWQ"` | ❌ OOM | Exceeded available memory; unable to run on an RTX 4090. |
| `vllm serve "ProbioticFarmer/phi-4-Q5_K_M-GGUF"` | ⚠️ Context-dependent | Can handle large context windows on 24 GB GPUs. **RTX 4090 consistently ran out of memory** under the tested configuration (observed repeatedly on 16 Jul 2026). |

---

## Higher VRAM GPU Testing

### Hardware

- **GPU pools tested:** H100 SXM, RTX 5090
- **Context window:** 128k tokens

### Log

```text
16/07/2026, 17:06:12
(Worker pid=541) INFO 07-16 11:36:12 [gpu_model_runner.py:5820] Encoder cache will be initialized with a budget of 128000 tokens, and profiled with 7 image items of the maximum feature size.
```

### Observation

> ⚠️ The configured **128k context window** was likely excessive and may have been the primary cause of the observed out-of-memory failures.

---

## Pod Inference (Ollama)

### Configuration

| Setting | Value |
|---|---|
| **Template** | `standard ollama/ollama:latest` |
| **Model** | `ollama run hf.co/ProbioticFarmer/phi-4-Q5_K_M-GGUF:Q5_K_M` |

## Result

- ✅ Works successfully.
- Very low cold-start time.
- More cost-effective than the alternative approaches tested.
