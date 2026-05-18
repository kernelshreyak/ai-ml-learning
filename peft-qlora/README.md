## PEFT QLoRA Reasoning Fine-Tuning

This project demonstrates a compact two-stage fine-tuning pipeline for a small reasoning model using:

- `PEFT` for low-rank adapter training
- `QLoRA` for 4-bit base model loading
- `Unsloth` for fast model loading and training
- `TRL` for both supervised fine-tuning and preference-style reinforcement learning

The code is centered around GSM8K math reasoning and uses a Qwen 2.5 1.5B instruct base model loaded in 4-bit precision.

### What This Folder Contains

- `train_reasoning_sft.py` - supervised fine-tuning on GSM8K with LoRA adapters
- `train_grpo.py` - GRPO refinement pass on the same base model plus the saved SFT adapter
- `finetuned_reasoning_inference.ipynb` - notebook for loading the adapter and running inference

### Training Pipeline

The workflow is intentionally split into two stages:

1. **SFT stage**
   - Loads `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`
   - Applies LoRA adapters to the attention and MLP projection layers
   - Trains on `gsm8k/main` using the first 5,000 training examples
   - Saves the adapter to `reasoning_adapter/`

2. **GRPO stage**
   - Reloads the same 4-bit base model
   - Re-attaches the `reasoning_adapter/` weights produced by SFT
   - Trains on the first 2,000 GSM8K examples with a reward function
   - Saves the refined adapter to `grpo_reasoning_adapter/`

### Supervised Fine-Tuning Details

The SFT script builds chat-formatted training rows from each GSM8K example:

- `system`: "You are a careful reasoning assistant."
- `user`: the GSM8K question
- `assistant`: the reference answer from the dataset

The formatted conversation is passed through `tokenizer.apply_chat_template(...)` and terminated with the tokenizer EOS token.

Important configuration values:

- `max_seq_length = 2048`
- `r = 8`
- `lora_alpha = 32`
- `lora_dropout = 0.05`
- `per_device_train_batch_size = 1`
- `gradient_accumulation_steps = 16`
- `num_train_epochs = 3`
- `learning_rate = 2e-4`
- `optim = "adamw_8bit"`
- `lr_scheduler_type = "cosine"`
- `output_dir = "outputs"`

LoRA is applied to:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

This covers both attention and MLP pathways, which is a common choice when adapting small instruction models for reasoning tasks.

### GRPO Details

The GRPO script reuses the SFT adapter and performs a second training pass with a reward signal.

Key choices:

- `max_seq_length = 1024`
- `num_generations = 4`
- `max_prompt_length = 256`
- `max_completion_length = 256`
- `learning_rate = 5e-6`
- `gradient_accumulation_steps = 4`
- `num_train_epochs = 1`
- `optim = "paged_adamw_8bit"`

The prompt format asks the model to:

- solve step by step
- end every answer with `#### <final_answer>`

The reward function extracts the text after the final `####` marker from both the generated completion and the ground-truth answer and assigns:

- `1.0` for exact match
- `0.0` otherwise

This makes the reward sparse but easy to interpret and keeps the training objective aligned with the final-answer format used in GSM8K.

### Inference Flow

The notebook shows the expected inference sequence:

1. Load the same 4-bit base model
2. Load the adapter from `reasoning_adapter/`
3. Switch the model to inference mode with `FastLanguageModel.for_inference(model)`
4. Generate completions with deterministic decoding settings such as:
   - `do_sample=False`
   - `repetition_penalty=1.15`
   - `max_new_tokens=128`

The notebook also includes a small prompt cleanup helper that trims text before instruction markers so the printed response is easier to read.

### Environment Requirements

You need a CUDA-capable GPU for practical training. The scripts assume:

- Python 3
- `torch`
- `unsloth`
- `trl`
- `datasets`
- `transformers` and related Hugging Face dependencies

The SFT and GRPO scripts use 4-bit loading and 8-bit optimizers to reduce VRAM usage, but the exact footprint still depends on GPU memory, sequence length, and batch configuration.

### Suggested Setup

From the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you are working on this folder independently, make sure your environment includes the packages required by `unsloth` and `trl`.

### Running SFT

```bash
cd peft-qlora
python train_reasoning_sft.py
```

Outputs:

- `outputs/` - trainer logs and checkpoints
- `reasoning_adapter/` - saved LoRA adapter and tokenizer files

### Running GRPO

Run the SFT stage first so `reasoning_adapter/` exists.

```bash
cd peft-qlora
python train_grpo.py
```

Outputs:

- `grpo_outputs/` - GRPO training logs and checkpoints
- `grpo_reasoning_adapter/` - refined adapter and tokenizer files

### Notes On Data And Reproducibility

- Both scripts use the public `gsm8k` dataset from Hugging Face.
- The SFT script trains on `train[:5000]`.
- The GRPO script trains on `train[:2000]`.
- Random seed in SFT is set to `3407`.
- The GRPO script uses `torch.cuda.is_bf16_supported()` to choose bf16 when available.

Because the model is trained with adapter-based updates only, the base 4-bit checkpoint remains unchanged. That makes it easy to reuse the same base model with different adapters for experiments.

### Common Failure Modes

- Missing CUDA or unsupported GPU memory: reduce sequence length, batch size, or accumulation steps.
- Adapter load errors: confirm `reasoning_adapter/` exists before starting GRPO or notebook inference.
- Dataset download failures: verify network access to Hugging Face Hub.
- Generation does not stop cleanly: make sure the prompt format still ends with the expected answer tag and EOS handling is preserved.

### Practical Extension Ideas

- Swap GSM8K for another reasoning dataset with the same chat formatting pipeline.
- Change the reward function to support partial credit instead of exact match only.
- Save and compare SFT vs GRPO adapters to evaluate whether reinforcement refinement improves answer formatting or correctness.
- Increase `max_seq_length` or the LoRA rank if you have more VRAM and want a larger adaptation capacity.
