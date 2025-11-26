import M1_alg as alg
import M1_statistics as frw
import time 

h, w = 112, 92
A, B, labels_A, labels_B = alg.constructTrainingMatrix(40, 10)

nn_count_correct = {1: 0, 2: 0, 3: 0}
knn_count_correct = {(norm, k): 0 for norm in [1, 2, 3] for k in [3, 5, 7]}

nn_timetaken_array = {norm: [] for norm in [1, 2, 3]}
knn_timetaken_array = {(norm, k): [] for norm in [1, 2, 3] for k in [3, 5, 7]}

person_num = B.shape[1]
start = time.time()

for i in range(person_num):
    vectorized_img = B[:, i]
    poza_test = vectorized_img.reshape(h, w)

    for norm in [1, 2, 3]:
        for k in [3, 5, 7]:
            label, io, knnTime = alg.kNN(vectorized_img, A, labels_A, norm, k)
            knn_timetaken_array[(norm, k)].append(knnTime)
            if label == labels_B[i]:
                knn_count_correct[(norm, k)] += 1

    for norm in [1, 2, 3]:
        idx, nnTime = alg.NN(vectorized_img, A, norm)
        nn_timetaken_array[norm].append(nnTime)
        if labels_A[idx] == labels_B[i]:
            nn_count_correct[norm] += 1

end = time.time()
timpul_de_executie = end - start

nn_rate = {norm: nn_count_correct[norm] / person_num for norm in [1, 2, 3]}
knn_rate = {(norm, k): knn_count_correct[(norm, k)] / person_num for norm in [1, 2, 3] for k in [3, 5, 7]}

print("┌ STATISTICA")
print(f"├ Timpul total de executare pentru {person_num} persoane: {timpul_de_executie:.4f}s")
print(f"├ Timpul mediu per imagine: {timpul_de_executie/person_num:.4f}s")
print(f"├  Matricea de antrenare conține: {400-person_num} imagini")
print(f"├  Matricea de test conține: {person_num} imagini")
print("├─────────── Rezultate NN ───────────")
for norm in [1, 2, 3]:
    print(f"│ Norma {norm}: {nn_rate[norm]*100:.2f}%")
print("├─────────── Rezultate kNN ───────────")
for norm in [1, 2, 3]:
    for k in [3, 5, 7]:
        print(f"│ Norma {norm}, k={k}: {knn_rate[(norm, k)]*100:.2f}%")
print("└─────────────────────────────────────")

frw.plot_execution_times_dict(nn_timetaken_array, knn_timetaken_array)
frw.plot_recognition_rates(nn_rate, knn_rate, "Comparatia Ratelor de Recunoastere")
