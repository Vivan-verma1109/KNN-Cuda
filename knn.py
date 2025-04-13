from numba import cuda
cuda.select_device(0)
cuda.detect()

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
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

def predict_one(test_point, k = 11):
    n_train = X_train.shape[0]     # Number of training samples
    x_train_flat = X_train.flatten() #flatten so we can use the kernel code I think
    
    
    # Allocate GPU memory for:
    # - the flattened training set
    # - the test point
    # - the output distance array (one float per training point)
    x_train_gpu = cuda.mem_alloc(x_train_flat.nbytes)
    x_test_gpu = cuda.mem_alloc(test_point.nbytes)
    dist_out_gpu = cuda.mem_alloc(n_train * np.float32().nbytes)
     
     
    # Transfer data from CPU (host) to GPU (device) 
    cuda.memcpy_htod(x_train_gpu, x_train_flat) # these are memcpys from cpu to gpu I think
    cuda.memcpy_htod(x_test_gpu, test_point)

    # Configure thread layout: 1D grid of blocks, each with 256 threads
    threads = 256
    
    #this that ceiling division shit
    blocks = (n_train + threads - 1) // threads
    
    # Launch the CUDA kernel to compute distances
    compute_distances(
        x_train_gpu, x_test_gpu, dist_out_gpu, np.int32(n_train),
        block=(threads, 1, 1), grid=(blocks, 1)
    )
     # Allocate CPU-side array to hold distances, and copy result back
    dist_out = np.empty(n_train, dtype=np.float32)
    cuda.memcpy_dtoh(dist_out, dist_out_gpu)

    # Sort distances and get indices of the k nearest neighbors
    top_k_idx = np.argsort(dist_out)[:k]

    # Get the corresponding labels and vote for the most common one
    top_k_labels = y_train[top_k_idx]
    return Counter(top_k_labels).most_common(1)[0][0]

# Predict
y_pred = np.array([predict_one(x) for x in X_test])

# Results
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
    