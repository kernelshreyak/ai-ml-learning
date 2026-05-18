from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

import re
import torch


# -----------------------------
# CONFIG
# -----------------------------

max_seq_length = 1024

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"

LORA_PATH = "reasoning_adapter"


# -----------------------------
# LOAD MODEL
# -----------------------------

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

# Re-attach SAME LoRA adapter
# trained during SFT.
model.load_adapter(LORA_PATH)

# Enable fast inference kernels.
FastLanguageModel.for_inference(model)


# -----------------------------
# LOAD DATASET
# -----------------------------

dataset = load_dataset(
    "gsm8k",
    "main",
    split="train[:2000]",
)


# -----------------------------
# FORMAT DATA
# -----------------------------

SYSTEM_PROMPT = """
You are a reasoning assistant.

Solve step-by-step.

End every answer with:
#### <final_answer>
"""


def make_prompt(example):
    return {
        "prompt": (
            SYSTEM_PROMPT
            + "\n\nQuestion:\n"
            + example["question"]
        ),
        "answer": example["answer"],
    }


dataset = dataset.map(make_prompt)


# -----------------------------
# REWARD FUNCTION
# -----------------------------

def extract_final_answer(text):
    """
    Extract:
    #### 42
    """

    match = re.search(r"####\s*(.*)", text)

    if match:
        return match.group(1).strip()

    return None


def reward_function(prompts, completions, answer, **kwargs):
    rewards = []

    for completion, gt in zip(completions, answer):

        generated_text = completion[0]["content"]

        pred = extract_final_answer(generated_text)
        gt_ans = extract_final_answer(gt)

        # Exact-match reward
        if pred == gt_ans:
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


# -----------------------------
# GRPO TRAINER
# -----------------------------

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,

    reward_funcs=reward_function,

    train_dataset=dataset,

    args=GRPOConfig(
        output_dir="grpo_outputs",

        learning_rate=5e-6,

        per_device_train_batch_size=1,

        gradient_accumulation_steps=4,

        num_generations=4,

        max_prompt_length=256,
        max_completion_length=256,

        num_train_epochs=1,

        logging_steps=1,

        save_steps=50,

        bf16=torch.cuda.is_bf16_supported(),

        optim="paged_adamw_8bit",
    ),
)

trainer.train()


# Save refined RL adapter
model.save_pretrained("grpo_reasoning_adapter")
tokenizer.save_pretrained("grpo_reasoning_adapter")