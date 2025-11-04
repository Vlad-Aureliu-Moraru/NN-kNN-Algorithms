import numpy as np

def append_results(nn_array, knn_array, filename="results.txt"):
    nn_results = []
    knn_results = []

    try:
        with open(filename, "r") as f:
            lines = [line.strip() for line in f.readlines()]
            mode = None
            for line in lines:
                if line == "NN":
                    mode = "NN"
                elif line == "KNN":
                    mode = "KNN"
                elif line:
                    values = [x == "True" for x in line.split()]
                    if mode == "NN":
                        nn_results.extend(values)
                    elif mode == "KNN":
                        knn_results.extend(values)
    except FileNotFoundError:
        pass  # No previous data — start fresh

    nn_results.extend(nn_array)
    knn_results.extend(knn_array)

    with open(filename, "w") as f:
        f.write("NN\n")
        f.write(" ".join(map(str, nn_results)) + "\n")
        f.write("KNN\n")
        f.write(" ".join(map(str, knn_results)) + "\n")



def read_and_average(filename="results.txt"):
    nn_results = []
    knn_results = []

    try:
        with open(filename, "r") as f:
            lines = [line.strip() for line in f.readlines()]
            mode = None
            for line in lines:
                if line == "NN":
                    mode = "NN"
                elif line == "KNN":
                    mode = "KNN"
                elif line:
                    values = [x == "True" for x in line.split()]
                    if mode == "NN":
                        nn_results.extend(values)
                    elif mode == "KNN":
                        knn_results.extend(values)
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return 0, 0

    if not nn_results or not knn_results:
        return 0, 0

    avg_nn = int(sum(nn_results) / len(nn_results) * 100)
    avg_knn = int(sum(knn_results) / len(knn_results) * 100)

    return avg_nn, avg_knn


