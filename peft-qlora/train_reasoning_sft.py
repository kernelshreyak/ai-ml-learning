# Import Unsloth's optimized model wrapper.
# FastLanguageModel gives faster loading, training, and inference for supported models.
from unsloth import FastLanguageModel

# Hugging Face datasets utility.
# We use this to download/load GSM8K, a grade-school math reasoning dataset.
from datasets import load_dataset

# TRL's supervised fine-tuning trainer.
# SFTTrainer trains the model to imitate target responses using next-token prediction.
from trl import SFTTrainer, SFTConfig


# Maximum context length used during training.
# 2048 means each training example can contain up to 2048 tokens.
# Larger values allow longer reasoning traces but require more VRAM.
max_seq_length = 2048


# Load the base model and tokenizer.
# This is a 1.5B parameter Qwen instruct model, already quantized to 4-bit.
model, tokenizer = FastLanguageModel.from_pretrained(
    # Unsloth-provided 4-bit version of Qwen2.5-1.5B-Instruct.
    model_name="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",

    # Tell Unsloth the maximum sequence length we plan to train with.
    max_seq_length=max_seq_length,

    # This enables QLoRA-style loading:
    # the frozen base model weights are loaded in 4-bit precision.
    load_in_4bit=True,
)


# Add LoRA adapters to the base model.
# The original model weights stay frozen.
# Only the small LoRA adapter matrices are trained.
model = FastLanguageModel.get_peft_model(
    model,

    # LoRA rank.
    # Higher r = more trainable capacity, but more VRAM and slower training.
    # r=16 is a good starting point for small reasoning finetunes.
    r=8,

    # These are the transformer linear layers where LoRA adapters are inserted.
    # Attention projections:
    # q_proj: query projection
    # k_proj: key projection
    # v_proj: value projection
    # o_proj: output projection
    #
    # MLP projections:
    # gate_proj, up_proj, down_proj
    #
    # Including both attention and MLP layers gives better reasoning adaptation
    # than adapting attention only.
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],

    # LoRA scaling factor.
    # Effective LoRA update is usually scaled by alpha / r.
    # Here: 32 / 16 = 2.
    lora_alpha=32,

    # Dropout applied inside LoRA adapters.
    # 0.05 can slightly reduce overfitting.
    # For tiny clean datasets, 0.0 or 0.05 are both common.
    lora_dropout=0.05,

    # Do not train bias terms.
    # This keeps training smaller and more stable.
    bias="none",

    # Saves VRAM by recomputing some activations during backward pass.
    # "unsloth" uses Unsloth's optimized gradient checkpointing.
    use_gradient_checkpointing="unsloth",
)


# Load the GSM8K training split.
# GSM8K examples contain:
# - question: math word problem
# - answer: step-by-step solution plus final answer
#
# train[:5000] uses only the first 5000 examples for a quick first run.
dataset = load_dataset("gsm8k", "main", split="train[:5000]")


# Convert each GSM8K row into a single training text string.
# SFTTrainer expects examples like:
#
# prompt + desired assistant answer
#
# During training, the model learns to predict the next token
# throughout this full text.
def formatting_func(example):

    messages = [
        {
            "role": "system",
            "content": "You are a careful reasoning assistant."
        },
        {
            "role": "user",
            "content": example["question"]
        },
        {
            "role": "assistant",
            "content": example["answer"]
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    text += tokenizer.eos_token

    return {"text": text}

# Apply formatting_func to every dataset row.
# After this, each row has a "text" field used for training.
dataset = dataset.map(formatting_func)


# Create the supervised fine-tuning trainer.
trainer = SFTTrainer(
    # The LoRA-wrapped model.
    model=model,

    # Tokenizer corresponding to the base model.
    tokenizer=tokenizer,

    # Dataset containing the formatted "text" field.
    train_dataset=dataset,

    # Training configuration.
    args=SFTConfig(
        # Name of the dataset column containing full training text.
        dataset_text_field="text",

        # Maximum number of tokens per training sample.
        # Longer examples are truncated.
        max_seq_length=max_seq_length,

        # Number of examples processed per GPU at once.
        # Keep low if VRAM is limited.
        per_device_train_batch_size=1,

        # Accumulate gradients for 4 forward/backward passes before optimizer step.
        # Effective batch size = 2 * 4 = 8 examples per optimizer update.
        gradient_accumulation_steps=16,

        # Number of warmup optimizer steps.
        # Learning rate gradually increases during warmup.
        warmup_steps=10,

        # One epoch means one pass over the selected 5000 examples.
        num_train_epochs=3,

        # Learning rate for LoRA adapter training.
        # LoRA commonly uses higher LR than full finetuning.
        learning_rate=2e-4,

        # Print training loss every step.
        logging_steps=1,

        # 8-bit AdamW optimizer.
        # Saves VRAM compared to normal AdamW.
        optim="adamw_8bit",

        # L2 regularization.
        # Helps prevent overfitting.
        weight_decay=0.01,

        # Cosine learning-rate decay after warmup.
        lr_scheduler_type="cosine",

        # Random seed for reproducibility.
        seed=3407,

        # Directory where checkpoints/logs are written.
        output_dir="outputs",
    ),
)


# Start training.
# This optimizes only the LoRA adapter weights.
# The frozen 4-bit base model is not directly updated.
trainer.train()


# Save the trained LoRA adapter.
# This directory will contain adapter_model.safetensors and adapter config.
model.save_pretrained("reasoning_adapter")

# Save tokenizer files so the adapter directory is self-contained enough
# for later loading/testing.
tokenizer.save_pretrained("reasoning_adapter")