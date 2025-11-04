import algorithm_related as alg
import fileResultsWriter as frw
import time 
h, w = 112, 92


A,B,labels_A,labels_B = alg.constructTrainingMatrix(40,8)

nn_count_correct = 0
knn_count_correct = 0

nn_result_array = []
knn_result_array = []

person_num = (B.size/10304)
start = time.time()
for i in range (int(person_num)):
    vectorized_img = B[:,i]
    poza_test = vectorized_img.reshape(h, w)
    label,io = alg.kNN(vectorized_img, A,labels_A, 2, 3)  
    knn_result = A[:, io].reshape(h, w)
    if labels_A[io] == labels_B[i]:
        knn_result_array.append(True)
        knn_count_correct+=1
    else:
        knn_result_array.append(False)
    i1 =alg.NN(vectorized_img, A, 2)       
    nn_result = A[:, i1].reshape(h, w)
    if labels_A[i1] == labels_B[i]:
        nn_result_array.append(True)
        nn_count_correct+=1
    else:
        nn_result_array.append(False)
    
end = time.time()
frw.append_results(nn_result_array,knn_result_array)
total_avg_nn,total_avg_knn = frw.read_and_average()
timpul_de_executie = end-start
print("---=====STATISTICA=====---")
print(f"• Rata Totala De Recunoasteri (%) KNN {total_avg_knn}%")
print(f"• Rata Totala De Recunoasteri (%)  NN {total_avg_nn}%")
print(f"• Timpul de executare al algoritmului pentru {person_num:.0f} persoane {timpul_de_executie:.4f}s")
print(f"• Timpul Aproximat De Executie Pentru Procesarea Unei Imagini {timpul_de_executie/person_num}s")
print(f"• Matricea De Antrenare Contine : {320-person_num:.0f} imagini ")
print(f"• Rata de Recunoastere Pentru Acest Run (NN): {nn_count_correct/(320-person_num)}")
print(f"• Rata de Recunoastere Pentru Acest Run (kNN): {knn_count_correct/(320-person_num)}")

#alg.displayImages(poza_test,knn_result,nn_result)


