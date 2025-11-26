import M2_helper as m2h

k_vals = [20,40]
output_file = "lanczos_results2.txt"


# Open file to write
with open(output_file, "w") as f:
    f.write("K\tTraining Images\tTesting Images\tAccuracy\tAvg Recognition Time(s)\tPreprocessing Time(s)\n")

    for K in k_vals:
        A_size, B_size, accuracy, avg_time, preprocessing_time = m2h.run_lanczos("80%", K)
        
        # Write results to file
        f.write(f"{K}\t{A_size}\t{B_size}\t{accuracy*100:.2f}\t{avg_time:.6f}\t{preprocessing_time:.6f}\n")
        
        print(f"K={K}: Accuracy={accuracy*100:.2f}%, Avg time={avg_time:.6f}s, Preprocessing={preprocessing_time:.6f}s")
