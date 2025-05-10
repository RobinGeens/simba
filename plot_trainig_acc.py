import re

import matplotlib.pyplot as plt


def parse_log_file(filename):
    epochs = []
    acc1 = []
    acc5 = []

    with open(filename, "r") as f:
        for line in f:
            # Look for lines with test accuracy summary
            if "* Acc@1" in line:
                # Extract accuracies using regex
                acc1_match = re.search(r"Acc@1 (\d+\.\d+)", line)
                acc5_match = re.search(r"Acc@5 (\d+\.\d+)", line)

                if acc1_match and acc5_match:
                    acc1.append(float(acc1_match.group(1)))
                    acc5.append(float(acc5_match.group(1)))
                    epochs.append(len(acc1))

    return epochs, acc1, acc5


# Parse both log files
simba_s_epochs, simba_s_acc1, simba_s_acc5 = parse_log_file("checkpoints/simba_s/log.txt")
simba_l_epochs, simba_l_acc1, simba_l_acc5 = parse_log_file("nohup.out")

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
