import json
from argparse import ArgumentParser

import matplotlib.pyplot as plt


def _parse_args():
    parser = ArgumentParser(
        description="Plot training and validation loss curves from a "
        "trainer_state.json file."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a trainer_state.json file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="loss_curve.png",
        help="Path to save the loss curve plot.",
    )
    return parser.parse_args()


def _load_history(input_path):
    with open(input_path) as f:
        state = json.load(f)
    return state["log_history"]


def main(args):
    log_history = _load_history(args.input)

    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []
    for entry in log_history:
        if "loss" in entry:
            train_steps.append(entry["step"])
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_loss.append(entry["eval_loss"])

    plt.figure(figsize=(8, 5))
    if train_loss:
        plt.plot(train_steps, train_loss, marker="o", label="train loss")
    if eval_loss:
        plt.plot(eval_steps, eval_loss, marker="s", label="val loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved loss curve to {args.output}")


if __name__ == "__main__":
    main(_parse_args())
