from llm_finetune.config import (
    MODEL_NAME,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    MAX_SEQ_LENGTH,
)
from peft import LoraConfig
from transformers import AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from llm_finetune.data import get_dataset
from llm_finetune.utils import tokenizer, device

base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "v_proj"],
)

train_dataset = get_dataset(split="train")
val_dataset = get_dataset(split="val")

training_args = SFTConfig(
    output_dir="././sft_output",
    max_steps=20,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=20,
    eval_strategy="steps",
    eval_steps=20,
    max_length=MAX_SEQ_LENGTH,
)

trainer = SFTTrainer(
    model=base_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=lora_config,
    processing_class=tokenizer,
)


trainer.train()
