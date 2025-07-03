import matplotlib.pyplot as plt
import seaborn

log_files = [
    {"path": "checkpoints/simba_l_bf16_B/tlog.txt", "label": "Simba-L-BF16"},
    {"path": "checkpoints/simba_l_FP32_B/tlog.txt", "label": "Simba-L-FP32"},
]


def parse_log_file(filename):
    epochs = []
    acc1 = []
    acc5 = []

    with open(filename, "r") as f:
        for line in f:
            try:
                # Parse JSON-formatted log lines
                data = eval(line)
                if "test_acc1" in data and "test_acc5" in data:
                    acc1.append(data["test_acc1"])
                    acc5.append(data["test_acc5"])
                    epochs.append(data["epoch"])
            except Exception:
                continue

    # Sort data points by epoch
    sorted_data = sorted(zip(epochs, acc1, acc5))
    epochs = [x[0] for x in sorted_data]
    acc1 = [x[1] for x in sorted_data]
    acc5 = [x[2] for x in sorted_data]

    return epochs, acc1, acc5


def make_plot():

    plt.figure(figsize=(10, 6))

    colors = seaborn.color_palette("muted")

    for i, config in enumerate(log_files):
        epochs, acc1, acc5 = parse_log_file(config["path"])

        plt.plot(epochs, acc1, color=colors[i], ls="-", linewidth=2)
        plt.plot(epochs, acc5, color=colors[i], ls="--", linewidth=2)

    plt.axhline(y=83.9, color="red", linestyle=":", linewidth=2, label="Target")

    for i, config in enumerate(log_files):
        plt.plot([], [], color=colors[i], label=f"{config['label']}")

    plt.plot([], [], color="gray", label="Top1", ls="-")
    plt.plot([], [], color="gray", label="Top5", ls="--")

    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Training Accuracy (%)", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_plot.png", bbox_inches="tight", pad_inches=0)
    plt.close()


if __name__ == "__main__":
    make_plot()
