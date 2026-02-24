import json
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from grpo.config import OUTPUTS_DIR


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def load_metrics(json_path="grpo_metrics.json"):
    """Load metrics from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def plot_metrics(metrics_data, smooth=False, output_prefix="grpo", verbose=False):
    """Plot training metrics and save to PNG files."""
    train_accuracies_avg = metrics_data.get("train_accuracies_avg", metrics_data.get("accuracies_avg", []))
    val_accuracies_avg = metrics_data.get("val_accuracies_avg", {})
    kl_avg = metrics_data["kl_avg"]

    window = 10
    x_offset = 0
    if smooth:
        if len(train_accuracies_avg) >= window:
            train_accuracies_avg = moving_average(train_accuracies_avg, window)
        if len(kl_avg) >= window:
            kl_avg = moving_average(kl_avg, window)
        x_offset = max(0, window - 1) if len(metrics_data.get("kl_avg", [])) >= window else 0

    # Plot 1: Accuracy
    fig, ax = plt.subplots()
    ax.set_xlabel("Step (x10)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)

    x_train_offset = x_offset if smooth and len(metrics_data.get("train_accuracies_avg", metrics_data.get("accuracies_avg", []))) >= window else 0
    x_main = range(x_train_offset, x_train_offset + len(train_accuracies_avg))
    ax.plot(x_main, train_accuracies_avg, color="tab:orange", label="Train Accuracy")

    # val_accuracies_avg can be a dict (new format) or a list (old format)
    if isinstance(val_accuracies_avg, dict):
        colors = {"3num_zs": "tab:blue", "3num_os": "tab:green", "natural": "tab:purple", "4num_zs": "tab:cyan", "4num_os": "tab:pink"}
        labels = {"3num_zs": "Val 3num ZS", "3num_os": "Val 3num OS", "natural": "Val Natural", "4num_zs": "Val 4num ZS", "4num_os": "Val 4num OS"}
        for key, vals in val_accuracies_avg.items():
            if len(vals) == 0:
                continue
            v = vals
            if smooth and len(v) >= window:
                v = moving_average(v, window)
            xv_offset = x_offset if smooth and len(vals) >= window else 0
            xv = range(xv_offset, xv_offset + len(v))
            ax.plot(xv, v, color=colors.get(key, "gray"), label=labels.get(key, key))
    elif isinstance(val_accuracies_avg, list) and len(val_accuracies_avg) > 0:
        v = val_accuracies_avg
        if smooth and len(v) >= window:
            v = moving_average(v, window)
        xv_offset = x_offset if smooth and len(val_accuracies_avg) >= window else 0
        xv = range(xv_offset, xv_offset + len(v))
        ax.plot(xv, v, color="tab:green", label="Val Accuracy")

    ax.set_title("Train/Val Accuracy{}".format(" (smoothed)" if smooth else " (20-step averages)"))
    ax.legend(loc="upper left", bbox_to_anchor=(0.05, 0.95), fontsize="small")

    curve_path = os.path.join(OUTPUTS_DIR, "{}_training_curve.png".format(output_prefix))
    fig.savefig(curve_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print("Saved: {}".format(curve_path))

    # Plot 2: KL Divergence
    fig_kl, ax_kl = plt.subplots()
    x_kl_offset = x_offset if smooth and len(metrics_data.get("kl_avg", [])) >= window else 0
    x_kl = range(x_kl_offset, x_kl_offset + len(kl_avg))
    ax_kl.plot(x_kl, kl_avg, color="tab:red", label="KL Divergence")
    ax_kl.set_xlabel("Step (x10)")
    ax_kl.set_ylabel("KL Divergence", color="tab:red")
    kl_max = max(kl_avg) if len(kl_avg) > 0 else 0
    ax_kl.set_ylim(0, kl_max * 1.1 if kl_max > 0 else 1)
    ax_kl.tick_params(axis='y', labelcolor="tab:red")
    ax_kl.set_title("KL Divergence over Training{}".format(" (smoothed)" if smooth else ""))
    ax_kl.legend(loc="upper left")
    kl_path = os.path.join(OUTPUTS_DIR, "{}_kl_divergence.png".format(output_prefix))
    fig_kl.savefig(kl_path, dpi=100, bbox_inches="tight")
    plt.close(fig_kl)
    if verbose:
        print("Saved: {}".format(kl_path))


def main():
    parser = argparse.ArgumentParser(description="Plot GRPO metrics with optional smoothing.")
    parser.add_argument('--smooth', action='store_true', help='Apply moving average smoothing (window=10)')
    parser.add_argument('--input', type=str, default="grpo_metrics.json", help='Path to metrics JSON file')
    parser.add_argument('--prefix', type=str, default=None, help='Output file prefix (default: derived from input)')
    args = parser.parse_args()

    metrics_data = load_metrics(args.input)
    prefix = args.prefix or args.input.replace("_metrics.json", "")
    plot_metrics(metrics_data, smooth=args.smooth, output_prefix=prefix, verbose=True)
    print("Plotting complete!")


if __name__ == "__main__":
    main()
