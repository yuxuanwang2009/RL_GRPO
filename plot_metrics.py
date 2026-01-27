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

def plot_metrics(metrics_data, smooth=False, output_prefix="grpo"):
    """Plot training metrics and save to PNG files."""
    rewards_avg = metrics_data["rewards_avg"]
    accuracies_avg = metrics_data["accuracies_avg"]
    kl_avg = metrics_data["kl_avg"]

    if smooth:
        window = 10
        rewards_avg = moving_average(rewards_avg, window)
        accuracies_avg = moving_average(accuracies_avg, window)
        kl_avg = moving_average(kl_avg, window)
        x_offset = window - 1
    else:
        x_offset = 0

    # Plot 1: Reward and Accuracy
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel("Step (x10)")
    ax1.set_ylabel("Reward", color="tab:blue")
    ax1.set_ylim(0, 1)
    ax2.yaxis.set_label_position('right')
    ax2.set_ylabel("Accuracy", color="tab:orange")
    ax2.set_ylim(0, 1)

    x = range(x_offset, x_offset + len(rewards_avg))
    ax1.plot(x, rewards_avg, color="tab:blue", label="Reward")
    ax2.plot(x, accuracies_avg, color="tab:orange", label="Accuracy")

    ax1.set_title("Reward and Accuracy{}".format(" (smoothed)" if smooth else " (20-step averages)"))

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", bbox_to_anchor=(0.05, 0.95))

    fig.savefig("{}_training_curve.png".format(output_prefix), dpi=100, bbox_inches="tight")
    print("Saved: {}_training_curve.png".format(output_prefix))

    # Plot 2: KL Divergence
    fig_kl, ax_kl = plt.subplots()
    ax_kl.plot(x, kl_avg, color="tab:red", label="KL Divergence")
    ax_kl.set_xlabel("Step (x20)")
    ax_kl.set_ylabel("KL Divergence", color="tab:red")
    ax_kl.set_ylim(0, 100)
    ax_kl.tick_params(axis='y', labelcolor="tab:red")
    ax_kl.set_title("KL Divergence over Training{}".format(" (smoothed)" if smooth else ""))
    ax_kl.legend(loc="upper left")
    fig_kl.savefig("{}_kl_divergence.png".format(output_prefix), dpi=100, bbox_inches="tight")
    print("Saved: {}_kl_divergence.png".format(output_prefix))

def main():
    parser = argparse.ArgumentParser(description="Plot GRPO metrics with optional smoothing.")
    parser.add_argument('--smooth', action='store_true', help='Apply moving average smoothing (window=10)')
    args = parser.parse_args()

    metrics_data = load_metrics()
    plot_metrics(metrics_data, smooth=args.smooth)
    print("Plotting complete!")

if __name__ == "__main__":
    main()