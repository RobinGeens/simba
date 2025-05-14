import matplotlib.pyplot as plt


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
            except:
                continue

    # Sort data points by epoch
    sorted_data = sorted(zip(epochs, acc1, acc5))
    epochs = [x[0] for x in sorted_data]
    acc1 = [x[1] for x in sorted_data]
    acc5 = [x[2] for x in sorted_data]

    return epochs, acc1, acc5


# Parse both log files
simba_s_epochs, simba_s_acc1, simba_s_acc5 = parse_log_file("checkpoints/simba_s/tlog.txt")
simba_l_epochs, simba_l_acc1, simba_l_acc5 = parse_log_file("checkpoints/simba_l/tlog.txt")
# simba_l_bf16_epochs, simba_l_bf16_acc1, simba_l_bf16_acc5 = parse_log_file("checkpoints/simba_l_bf16/tlog.txt")

# Create the plot
plt.figure(figsize=(10, 6))

# Plot top-1 accuracy
plt.plot(simba_s_epochs, simba_s_acc1, "b-", label="SimBA-S Top-1")
plt.plot(simba_l_epochs, simba_l_acc1, "r-", label="SimBA-L Top-1")

# Plot top-5 accuracy
plt.plot(simba_s_epochs, simba_s_acc5, "b--", label="SimBA-S Top-5")
plt.plot(simba_l_epochs, simba_l_acc5, "r--", label="SimBA-L Top-5")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training Accuracy vs Epoch")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_plot.png")
plt.close()
