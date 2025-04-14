from numba import cuda # type: ignore
cuda.select_device(0)
cuda.detect()

import pycuda.autoinit # type: ignore
import pycuda.driver as cuda # type: ignore
from pycuda.compiler import SourceModule # type: ignore
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from collections import Counter
# Load dataset
dataset = pd.read_csv('diabetes.csv')
zero_null = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for label in zero_null:
    dataset[label] = dataset[label].replace(0, np.nan)
    dataset[label] = dataset[label].fillna(dataset[label].mean())

X = dataset.iloc[:, :8].values.astype(np.float32)
y = dataset.iloc[:, 8].values.astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

kernel_code = '''
__global__ void compute_distances(float *x_train, float *x_test, float *dist_out, int n_train){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_train) return;
    
    float dist = 0;
    for(int i = 0; i < 8; i++){
        float diff = x_train[idx * 8 + i] - x_test[i];
        dist += diff * diff;
    }
    dist_out[idx] = dist;
}
'''

mod = SourceModule(kernel_code)
compute_distances = mod.get_function("compute_distances")

def predict_all(k=11):
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    D = X_train.shape[1]

    # Flatten
    x_train_flat = X_train.flatten()
    x_test_flat = X_test.flatten()

    # Allocate device memory
    x_train_gpu = cuda.mem_alloc(x_train_flat.nbytes)
    x_test_gpu = cuda.mem_alloc(x_test_flat.nbytes)
    dist_out_gpu = cuda.mem_alloc(n_test * n_train * np.float32().nbytes)

    # Copy to device
    cuda.memcpy_htod(x_train_gpu, x_train_flat)
    cuda.memcpy_htod(x_test_gpu, x_test_flat)

    # CUDA thread config
    threads_per_block = (16, 16)
    blocks_per_grid = (
        (n_train + threads_per_block[0] - 1) // threads_per_block[0],
        (n_test + threads_per_block[1] - 1) // threads_per_block[1]
    )

    # Kernel launch
    compute_distances(
        x_train_gpu,
        x_test_gpu,
        dist_out_gpu,
        np.int32(n_train),
        np.int32(n_test),
        np.int32(D),
        block=threads_per_block,
        grid=blocks_per_grid
    )

    # Copy distances back
    dist_out_host = np.empty((n_test * n_train), dtype=np.float32)
    cuda.memcpy_dtoh(dist_out_host, dist_out_gpu)

    # Reshape to matrix [n_test, n_train]
    dist_matrix = dist_out_host.reshape((n_test, n_train))

    # KNN voting
    y_pred = []
    for i in range(n_test):
        top_k_idx = np.argsort(dist_matrix[i])[:k]
        top_k_labels = y_train[top_k_idx]
        vote = Counter(top_k_labels).most_common(1)[0][0]
        y_pred.append(vote)

    return np.array(y_pred)



# Predict
y_pred = np.array(predict_all(k = 11))

# Results
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
    