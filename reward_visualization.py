import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import seaborn as sns
from pathlib import Path

def visualize_ppo_rewards(training_dir, output_dir=None):
    """
    Visualize rewards and other metrics from PPO training.
    
    Args:
        training_dir (str): Directory where training metrics are saved
        output_dir (str, optional): Directory to save visualizations. If None, uses training_dir
    """
    if output_dir is None:
        output_dir = training_dir
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load training metrics
    metrics_path = Path(training_dir) / "training_metrics.json"
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Load training info
    info_path = Path(training_dir) / "training_info.json"
    with open(info_path, 'r') as f:
        training_info = json.load(f)
    
    # Set style
    plt.style.use('ggplot')
    sns.set_palette("colorblind")
    
    # Create figure and subplots
    fig = plt.figure(figsize=(15, 20))
    fig.suptitle(f"PPO Training Metrics: {Path(training_dir).name}", fontsize=16)
    
    # 1. Reward Distribution
    ax1 = fig.add_subplot(3, 2, 1)
    sns.histplot(metrics["rewards"], bins=20, kde=True, ax=ax1)
    ax1.set_title("Reward Distribution")
    ax1.set_xlabel("Reward Value")
    ax1.set_ylabel("Frequency")
    
    # 2. Reward by Action
    rewards = np.array(metrics["rewards"])
    actions = np.array(metrics["actions"])
    
    # Create pandas DataFrame for easier analysis
    df = pd.DataFrame({
        "reward": rewards,
        "action": ["unsafe" if a == 1 else "safe" for a in actions]
    })
    
    ax2 = fig.add_subplot(3, 2, 2)
    sns.boxplot(x="action", y="reward", data=df, ax=ax2)
    ax2.set_title("Rewards by Classification")
    ax2.set_xlabel("Model Classification")
    ax2.set_ylabel("Reward Value")
    
    # 3. Reward Moving Average
    ax3 = fig.add_subplot(3, 2, 3)
    window_size = min(100, len(rewards) // 10) if len(rewards) > 10 else 1
    rolling_mean = pd.Series(rewards).rolling(window=window_size).mean()
    ax3.plot(rolling_mean)
    ax3.set_title(f"Reward Moving Average (Window={window_size})")
    ax3.set_xlabel("Sample")
    ax3.set_ylabel("Average Reward")
    
    # 4. Loss over time
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.plot(metrics["loss_values"])
    ax4.set_title("Loss Values")
    ax4.set_xlabel("Batch")
    ax4.set_ylabel("Loss")
    
    # 5. Action Distribution Over Time
    ax5 = fig.add_subplot(3, 2, 5)
    window_size = min(100, len(actions) // 10) if len(actions) > 10 else 1
    rolling_action = pd.Series(actions).rolling(window=window_size).mean()
    ax5.plot(rolling_action)
    ax5.set_title(f"Fraction of 'Unsafe' Predictions (Window={window_size})")
    ax5.set_xlabel("Sample")
    ax5.set_ylabel("Fraction of 'Unsafe'")
    ax5.set_ylim(-0.05, 1.05)
    
    # 6. PPO Ratios
    ax6 = fig.add_subplot(3, 2, 6)
    ratios = np.array(metrics["ratios"])
    # Filter out extreme outliers for better visualization
    ratios = ratios[ratios < np.percentile(ratios, 99)]
    sns.histplot(ratios, bins=30, kde=True, ax=ax6)
    ax6.axvline(x=1, color='k', linestyle='--')
    ax6.axvline(x=1+training_info["ppo_clip_eps"], color='r', linestyle='--')
    ax6.axvline(x=1-training_info["ppo_clip_eps"], color='r', linestyle='--')
    ax6.set_title("PPO Policy Update Ratios")
    ax6.set_xlabel("Ratio")
    ax6.set_ylabel("Frequency")
    
    # Add heatmap: Reward by epoch and action
    fig2 = plt.figure(figsize=(12, 8))
    fig2.suptitle("Reward Evolution Over Training", fontsize=16)
    
    # Create a new dataset with epochs
    df_epochs = pd.DataFrame({
        "reward": rewards,
        "action": ["unsafe" if a == 1 else "safe" for a in actions],
        "epoch": np.repeat(range(1, training_info["num_epochs"] + 1), 
                          len(rewards) // training_info["num_epochs"])[:len(rewards)]
    })
    
    # 7. Reward heatmap by epoch and action
    reward_pivot = df_epochs.pivot_table(
        values="reward", 
        index="epoch", 
        columns="action", 
        aggfunc="mean"
    )
    
    ax7 = fig2.add_subplot(1, 1, 1)
    sns.heatmap(reward_pivot, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax7)
    ax7.set_title("Average Reward by Epoch and Classification")
    
    # Save figures
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(Path(output_dir) / "ppo_reward_metrics.png", dpi=300)
    
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig(Path(output_dir) / "ppo_reward_heatmap.png", dpi=300)
    
    plt.close('all')
    
    # Analyze reward quality
    avg_reward = np.mean(rewards)
    safe_reward = df[df["action"] == "safe"]["reward"].mean()
    unsafe_reward = df[df["action"] == "unsafe"]["reward"].mean()
    
    # Create reward analysis summary
    analysis = {
        "average_reward": float(avg_reward),
        "safe_classification_avg_reward": float(safe_reward),
        "unsafe_classification_avg_reward": float(unsafe_reward),
        "reward_std": float(np.std(rewards)),
        "total_samples": len(rewards),
        "fraction_unsafe": float(np.mean(actions)),
        "reward_correlation_with_action": float(np.corrcoef(rewards, actions)[0, 1]),
    }
    
    # Save analysis
    with open(Path(output_dir) / "reward_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Visualizations saved to {output_dir}")
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Average reward for 'safe' classifications: {safe_reward:.4f}")
    print(f"Average reward for 'unsafe' classifications: {unsafe_reward:.4f}")
    
    return fig, fig2, analysis

# Example usage
if __name__ == "__main__":
    # Replace with your actual training directory
    visualize_ppo_rewards("ppo_safety_classifier")


# Advanced visualization: Comparing rewards to true labels (if available)
def analyze_reward_vs_true_labels(training_dir, csv_file, text_column, label_column, output_dir=None):
    """
    Compare reward model evaluations with true labels from the dataset.
    
    Args:
        training_dir (str): Directory where training metrics are saved
        csv_file (str): Path to the CSV file with original data and true labels
        text_column (str): Name of the column containing text data
        label_column (str): Name of the column containing true labels
        output_dir (str, optional): Directory to save visualizations
    """
    if output_dir is None:
        output_dir = training_dir
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load training metrics
    metrics_path = Path(training_dir) / "training_metrics.json"
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Load reward cache if available
    cache_path = Path(training_dir).parent / "reward_cache.json"
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            reward_cache = json.load(f)
    else:
        print("Reward cache not found. Cannot perform advanced analysis.")
        return None
    
    # Load dataset with true labels
    df_data = pd.read_csv(csv_file)
    
    if label_column not in df_data.columns:
        print(f"Label column '{label_column}' not found in dataset. Cannot perform comparison.")
        return None
    
    # Match texts with rewards from cache
    results = []
    
    for _, row in df_data.iterrows():
        text = row[text_column]
        true_label = int(row[label_column])
        
        # Check both possible actions in cache
        safe_key = f"{text}:0"
        unsafe_key = f"{text}:1"
        
        safe_reward = reward_cache.get(safe_key)
        unsafe_reward = reward_cache.get(unsafe_key)
        
        if safe_reward is not None and unsafe_reward is not None:
            results.append({
                "text": text[:100],  # Truncate for display
                "true_label": true_label,
                "true_class": "unsafe" if true_label == 1 else "safe",
                "reward_if_safe": safe_reward,
                "reward_if_unsafe": unsafe_reward,
                "reward_alignment": unsafe_reward > safe_reward if true_label == 1 else safe_reward > unsafe_reward
            })
    
    if not results:
        print("No matching texts found in reward cache. Cannot perform analysis.")
        return None
        
    df_results = pd.DataFrame(results)
    
    # Create visualization
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Reward Model Alignment with True Labels", fontsize=16)
    
    # 1. Distribution of rewards by true label
    ax1 = fig.add_subplot(2, 2, 1)
    df_melted = pd.melt(df_results, 
                        id_vars=["true_class"],
                        value_vars=["reward_if_safe", "reward_if_unsafe"],
                        var_name="action", 
                        value_name="reward")
    df_melted["action"] = df_melted["action"].map({
        "reward_if_safe": "safe", 
        "reward_if_unsafe": "unsafe"
    })
    
    sns.boxplot(x="true_class", y="reward", hue="action", data=df_melted, ax=ax1)
    ax1.set_title("Reward Distribution by True Class and Action")
    ax1.set_xlabel("True Class")
    ax1.set_ylabel("Reward")
    
    # 2. Reward alignment percentage
    ax2 = fig.add_subplot(2, 2, 2)
    alignment_rate = df_results["reward_alignment"].mean() * 100
    ax2.bar(["Aligned", "Misaligned"], 
            [alignment_rate, 100-alignment_rate],
            color=["green", "red"])
    ax2.set_title("Reward Model Alignment with True Labels")
    ax2.set_ylabel("Percentage")
    ax2.set_ylim(0, 100)
    for i, v in enumerate([alignment_rate, 100-alignment_rate]):
        ax2.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    # 3. Reward difference by true class
    df_results["reward_diff"] = df_results["reward_if_unsafe"] - df_results["reward_if_safe"]
    
    ax3 = fig.add_subplot(2, 2, 3)
    sns.histplot(data=df_results, x="reward_diff", hue="true_class", 
                 bins=20, kde=True, ax=ax3)
    ax3.axvline(x=0, color='k', linestyle='--')
    ax3.set_title("Reward Difference (Unsafe - Safe)")
    ax3.set_xlabel("Reward Difference")
    ax3.set_ylabel("Frequency")
    
    # 4. Confusion matrix of reward signals
    ax4 = fig.add_subplot(2, 2, 4)
    df_results["reward_prediction"] = df_results.apply(
        lambda x: 1 if x["reward_if_unsafe"] > x["reward_if_safe"] else 0, axis=1
    )
    
    # Create and plot confusion matrix
    conf_matrix = pd.crosstab(
        df_results["true_label"], 
        df_results["reward_prediction"],
        rownames=["True"],
        colnames=["Predicted"],
        normalize="index"
    )
    
    sns.heatmap(conf_matrix, annot=True, fmt=".1%", cmap="Blues", ax=ax4)
    ax4.set_title("Reward Model Prediction vs True Labels")
    
    # Save figure
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(Path(output_dir) / "reward_vs_true_labels.png", dpi=300)
    
    # Generate analysis report
    analysis = {
        "reward_alignment_rate": float(alignment_rate / 100),
        "average_reward_diff_true_safe": float(df_results[df_results["true_label"] == 0]["reward_diff"].mean()),
        "average_reward_diff_true_unsafe": float(df_results[df_results["true_label"] == 1]["reward_diff"].mean()),
        "reward_model_accuracy": float((df_results["true_label"] == df_results["reward_prediction"]).mean()),
        "samples_analyzed": len(df_results)
    }
    
    # Save analysis
    with open(Path(output_dir) / "reward_vs_true_labels_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Reward vs True Label analysis saved to {output_dir}")
    print(f"Reward alignment rate: {alignment_rate:.1f}%")
    print(f"Reward model accuracy: {analysis['reward_model_accuracy']*100:.1f}%")
    
    return fig, analysis

# Example usage of the advanced visualization
# analyze_reward_vs_true_labels(
#     "ppo_safety_classifier", 
#     "safety_dataset.csv", 
#     "content", 
#     "is_unsafe"
# )
