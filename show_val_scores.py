from matplotlib import pyplot as plt
import os
import argparse

def plot_loss(model_name):
    train_loss_file_path = os.path.join('runs', model_name, 'train_loss.txt')
    train_loss_plot_path = os.path.join('runs', model_name, 'train_loss_plot.png')

    steps = []
    losses = []

    with open(train_loss_file_path, 'r') as file:
        for line in file:
            step, loss = line.replace('\n', '').split(',')
            steps += [int(step) + 1]
            losses += [float(loss)]

    # Plot the loss
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label="Train loss", color="blue", linewidth=2)
    plt.xlabel("Step", fontsize=14)
    plt.ylabel("loss", fontsize=14)
    plt.title(f"Train loss {model_name}", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)

    # Save the plot to a file
    plt.savefig(train_loss_plot_path, dpi=300)
    print(f"Plot saved as '{train_loss_plot_path}'")

    # Show the plot (ensures it blocks execution in terminal)
    plt.show(block=True)



def plot_scores(model_name):
    # Path to your events file
    scores_file_path = os.path.join('runs', model_name, 'val_loss.txt')
    d1_plot_path = os.path.join('runs', model_name, 'val_d1_plot.png')
    epe_plot_path = os.path.join('runs', model_name, 'val_epe_plot.png')

    steps = []
    d1 = []
    epe = []
    
    with open(scores_file_path, 'r') as file:
        for line in file:
            step, score_name, value = line.replace('\n', '').split(',')
            if 'd1' in score_name:
                steps += [int(step) + 1]
                d1 += [float(value)]
            if 'epe' in score_name:
                epe += [float(value)]

    # Plot the d1
    plt.figure(figsize=(10, 6))
    plt.plot(steps, d1, label="Validation d1", color="blue", linewidth=2)
    plt.xlabel("Step", fontsize=14)
    plt.ylabel("d1", fontsize=14)
    plt.title(f"Validation d1 {model_name}", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)

    # Save the plot to a file
    plt.savefig(d1_plot_path, dpi=300)
    print(f"Plot saved as '{d1_plot_path}'")

    # Show the plot (ensures it blocks execution in terminal)
    plt.show(block=True)

    
    # Plot the epe
    plt.figure(figsize=(10, 6))
    plt.plot(steps, epe, label="Validation epe", color="blue", linewidth=2)
    plt.xlabel("Step", fontsize=14)
    plt.ylabel("epe", fontsize=14)
    plt.title(f"Validation epe {model_name}", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)

    # Save the plot to a file
    plt.savefig(epe_plot_path, dpi=300)
    print(f"Plot saved as '{epe_plot_path}'")

    # Show the plot (ensures it blocks execution in terminal)
    plt.show(block=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help="name of the model")
    args = parser.parse_args()
    plot_scores(args.model_name)
    plot_loss(args.model_name)