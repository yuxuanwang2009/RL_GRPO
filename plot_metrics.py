import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load metrics data
with open("grpo_metrics.json", "r") as f:
    metrics_data = json.load(f)

rewards_avg = metrics_data["rewards_avg"]
accuracies_avg = metrics_data["accuracies_avg"]
kl_avg = metrics_data["kl_avg"]

# Plot 1: Reward and Accuracy
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.set_xlabel("Step (x10)")
ax1.set_ylabel("Reward", color="tab:blue")
ax2.yaxis.set_label_position('right')
ax2.set_ylabel("Accuracy", color="tab:orange")

ax1.plot(rewards_avg, color="tab:blue", label="Reward (10-step avg)")
ax2.plot(accuracies_avg, color="tab:orange", label="Accuracy (10-step avg)")

ax1.set_title("Reward and Accuracy (10-step averages)")

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", bbox_to_anchor=(0.05, 0.95))

fig.savefig("grpo_training_curve.png", dpi=100, bbox_inches="tight")
print("Saved: grpo_training_curve.png")

# Plot 2: KL Divergence
fig_kl, ax_kl = plt.subplots()
ax_kl.plot(kl_avg, color="tab:red", label="KL Divergence (10-step avg)")
ax_kl.set_xlabel("Step (x10)")
ax_kl.set_ylabel("KL Divergence", color="tab:red")
ax_kl.tick_params(axis='y', labelcolor="tab:red")
ax_kl.set_title("KL Divergence over Training")
ax_kl.legend(loc="upper left")
fig_kl.savefig("grpo_kl_divergence.png", dpi=100, bbox_inches="tight")
print("Saved: grpo_kl_divergence.png")

print("Plotting complete!")
