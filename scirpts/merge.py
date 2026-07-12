from peft import AutoPeftModelForCausalLM
from argparse import ArgumentParser


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("--base-model-path", type=str, help="Path to the base model")
    parser.add_argument("--lora-model-path", type=str, help="Path to the LoRA model")
    parser.add_argument("--output-path", type=str, help="Path to save the merged model")
    return parser.parse_args()


def main(args):
    model = AutoPeftModelForCausalLM.from_pretrained(args.lora_model_path)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.output_path)


if __name__ == "__main__":
    args = _parse_args()
    main(args)
