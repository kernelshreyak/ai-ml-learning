# RunPod Testing Devlog

---

## 2026-07-15 — Initial Exploration

### Overview
Spent the day getting familiar with RunPod's interface and workflow.

### Findings

| Topic | Details |
|---|---|
| **Runpod Flash** | Explored the platform |
| **Interface** | Got accustomed to the general flow |
| **Volume performance** | Network volume is **much slower** than attached volume |
| **PyTorch experiments** | Ran tests via Jupyter Notebook on a pod |

---

## 2026-07-16 — Serverless Inference & GPU Benchmarks

### Serverless Setup
- [x] API key configured

### Fast Chat Models (16–24 GB VRAM)

| Command | Result | Notes |
|---|---|---|
| `vllm serve "openchat/openchat-3.5-0106"` | ✅ Works | Compatible with ChatBox & OpenAI SDK |
| `vllm serve "QuantTrio/Qwen3.5-27B-AWQ"` | ❌ OOM | Cannot run on RTX 4090 |
| `vllm serve "ProbioticFarmer/phi-4-Q5_K_M-GGUF"` | ⚠️ Context-dependent | Handles large context windows on 24 GB; **RTX 4090 = OOM** (observed repeatedly on 16 Jul 2026) |

### 80 GB GPU Tests

- **Pools tried:** H100 SXM, RTX 5090
- **Context window:** 128k

```
16/07/2026, 17:06:12
(Worker pid=541) INFO 07-16 11:36:12 [gpu_model_runner.py:5820] Encoder cache will be initialized with a budget of 128000 tokens, and profiled with 7 image items of the maximum feature size.
```

> ⚠️ The 128k context window was likely excessive — possibly caused OOM.

---

## 2026-07-17 — Pod Inference (Ollama)

### Configuration

| Setting | Value |
|---|---|
| **Template** | `standard ollama/ollama:latest` |
| **Model** | `llama run hf.co/ProbioticFarmer/phi-4-Q5_K_M-GGUF:Q5_K_M` |

### Result
- ✅ Works fine
- Very low cold start time
- Saves money compared to alternatives
