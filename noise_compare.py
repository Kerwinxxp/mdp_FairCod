import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Load JSON results
def load_results(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Load original and noisy results
results_original = load_results('results_experiment_noise_fee_1.json')
noise_levels = [1, 3,  5]
results_noisy = {level: load_results(f'results_experiment_noise_fee_{level}.json') for level in noise_levels}

# Extract days and metrics
days = list(results_original.keys())
metrics = ["meanReward", "meanEff", "varEff", "overdueRate"]

# Convert data into lists for plotting
original_values = {metric: [results_original[day][metric] for day in days] for metric in metrics}
noisy_values = {level: {metric: [results_noisy[level][day][metric] for day in days] for metric in metrics} for level in noise_levels}

# Create a PDF file to save the plots
with PdfPages('comparison_plots.pdf') as pdf:
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()

    for i, metric in enumerate(metrics):
        ax = axs[i]
        ax.plot(days, original_values[metric], marker='o', linestyle='-', label="Original", color='b')
        for level in noise_levels:
            ax.plot(days, noisy_values[level][metric], marker='s', linestyle='--', label=f"Noisy {level}", alpha=0.7)
        
        ax.set_xlabel("Days")
        ax.set_ylabel(metric)
        ax.set_title(f"Trend of {metric} Over Days")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xticks(days)
        ax.set_xticklabels(days, rotation=45)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print("Comparison plots saved to comparison_plots.pdf")