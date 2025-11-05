import matplotlib.pyplot as plt

def plot_recognition_rates(nn_rates, knn_rates, title="Recognition Rate Comparison"):
    labels = []
    values = []
    for norm, rate in nn_rates.items():
        labels.append(f"NN (norm={norm})")
        values.append(rate * 100)

    for (norm, k), rate in knn_rates.items():
        labels.append(f"kNN (norm={norm}, k={k})")
        values.append(rate * 100)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color=['#4CAF50' if 'NN' in l else '#2196F3' for l in labels])

    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Recognition Rate (%)")
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.2f}%", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_execution_times_dict(nn_times, knn_times):
    labels = []
    averages = []

    for norm, times in nn_times.items():
        labels.append(f"NN (norm={norm})")
        averages.append(sum(times)/len(times))

    for (norm, k), times in knn_times.items():
        labels.append(f"kNN (norm={norm}, k={k})")
        averages.append(sum(times)/len(times))

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, averages)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Timpul mediu  de executie (s)")
    plt.title("Timpul De Executie pe Algoritm  / Norm / k")
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    for bar, avg in zip(bars, averages):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{avg:.4f}s", 
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()
