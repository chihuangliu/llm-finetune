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
from argparse import ArgumentParser


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


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default="././sft_output",
        help="Directory to save the model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size per device during training.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Number of steps between logging.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=10,
        help="Number of steps between saving the model.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=10,
        help="Number of steps between evaluations.",
    )
    return parser.parse_args()


def main(args):
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=5e-5,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
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


if __name__ == "__main__":
    args = _parse_args()
    main(args)
