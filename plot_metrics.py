import json
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def load_metrics(json_path="grpo_metrics.json"):
    """Load metrics from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)

def plot_metrics(metrics_data, smooth=False, output_prefix="grpo", verbose=False):
    """Plot training metrics and save to PNG files."""
    if "train_accuracies_avg" in metrics_data and "val_accuracies_avg" in metrics_data:
        train_accuracies_avg = metrics_data["train_accuracies_avg"]
        val_accuracies_avg = metrics_data["val_accuracies_avg"]
    else:
        # Backward compatibility with older metrics format
        train_accuracies_avg = metrics_data["accuracies_avg"]
        val_accuracies_avg = metrics_data["accuracies_avg"]
    kl_avg = metrics_data["kl_avg"]

    if smooth:
        window = 10
        train_accuracies_avg = moving_average(train_accuracies_avg, window)
        kl_avg = moving_average(kl_avg, window)
        x_offset = window - 1
    else:
        x_offset = 0

    # Plot 1: Accuracy
    fig, ax = plt.subplots()
    ax.set_xlabel("Step (x10)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)

    x_main = range(x_offset, x_offset + len(train_accuracies_avg))
    x_val = range(len(val_accuracies_avg))
    ax.plot(x_main, train_accuracies_avg, color="tab:orange", label="Train Accuracy")
    ax.plot(x_val, val_accuracies_avg, color="tab:green", label="Val Accuracy")

    ax.set_title("Train/Val Accuracy{}".format(" (train smoothed)" if smooth else " (20-step averages)"))
    ax.legend(loc="upper left", bbox_to_anchor=(0.05, 0.95))

    fig.savefig("{}_training_curve.png".format(output_prefix), dpi=100, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print("Saved: {}_training_curve.png".format(output_prefix))

    # Plot 2: KL Divergence
    fig_kl, ax_kl = plt.subplots()
    ax_kl.plot(x_main, kl_avg, color="tab:red", label="KL Divergence")
    ax_kl.set_xlabel("Step (x10)")
    ax_kl.set_ylabel("KL Divergence", color="tab:red")
    ax_kl.set_ylim(0, 4)
    ax_kl.tick_params(axis='y', labelcolor="tab:red")
    ax_kl.set_title("KL Divergence over Training{}".format(" (smoothed)" if smooth else ""))
    ax_kl.legend(loc="upper left")
    fig_kl.savefig("{}_kl_divergence.png".format(output_prefix), dpi=100, bbox_inches="tight")
    plt.close(fig_kl)
    if verbose:
        print("Saved: {}_kl_divergence.png".format(output_prefix))

def main():
    parser = argparse.ArgumentParser(description="Plot GRPO metrics with optional smoothing.")
    parser.add_argument('--smooth', action='store_true', help='Apply moving average smoothing (window=10)')
    args = parser.parse_args()

    metrics_data = load_metrics()
    plot_metrics(metrics_data, smooth=args.smooth, verbose=True)
    print("Plotting complete!")

if __name__ == "__main__":
    main()