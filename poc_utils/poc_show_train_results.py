import tensorflow as tf
from matplotlib import pyplot as plt
import os
import argparse

def plot_loss(run_file_path, model_name):
    # Extract losses and steps
    steps = []
    losses = []
    for e in tf.compat.v1.train.summary_iterator(run_file_path):
        for v in e.summary.value:
            if "loss" in v.tag:  # Adjust the tag if necessary
                steps.append(e.step)
                losses.append(v.simple_value)

    # Plot the losses
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label="Training Loss", color="blue", linewidth=2)
    plt.xlabel("Step", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title(f"Training Loss {model_name}", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)

    # Save the plot to a file
    os.makedirs("loss_plots", exist_ok=True)
    plt.savefig(f"loss_plots/{model_name}_loss_plot.png", dpi=300)
    print(f"Plot saved as 'loss_plots/{model_name}_loss_plot.png'")

    # Show the plot (ensures it blocks execution in terminal)
    plt.show(block=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_file', help="path of run file to plot")
    parser.add_argument('--model_name', help="name of the model")
    args = parser.parse_args()
    plot_loss(args.run_file, args.model_name)
