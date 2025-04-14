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
// CUDA kernel to compute squared Euclidean distances between all test and train points
__global__ void compute_distances(float *x_train, float *x_test, float *dist_out, int n_train, int n_test, int D) {
    // Get indices this thread is responsible for
    int train_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int test_idx  = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check to avoid out-of-bounds memory access
    if (train_idx >= n_train || test_idx >= n_test) return;

    float dist = 0;
    // Loop over each feature dimension
    for (int d = 0; d < D; d++) {
        float diff = x_train[train_idx * D + d] - x_test[test_idx * D + d];
        dist += diff * diff; // accumulate squared differences
    }

    // Store computed squared distance in output matrix (row-major order)
    dist_out[test_idx * n_train + train_idx] = dist;
}
'''

# Compile the CUDA kernel
mod = SourceModule(kernel_code)

# Retrieve the kernel function by name
compute_distances = mod.get_function("compute_distances")

# KNN prediction using GPU for distance computation
def predict_all(k=11):
    n_train = X_train.shape[0]  # Number of training points
    n_test = X_test.shape[0]    # Number of test points
    D = X_train.shape[1]        # Number of features (e.g., 8)

    # Flatten the data for CUDA (row-major order)
    x_train_flat = X_train.flatten()
    x_test_flat = X_test.flatten()

    # Allocate memory on GPU
    x_train_gpu = cuda.mem_alloc(x_train_flat.nbytes)
    x_test_gpu = cuda.mem_alloc(x_test_flat.nbytes)
    dist_out_gpu = cuda.mem_alloc(n_test * n_train * np.float32().nbytes)

    # Copy data to GPU
    cuda.memcpy_htod(x_train_gpu, x_train_flat)
    cuda.memcpy_htod(x_test_gpu, x_test_flat)

    # Set up CUDA thread configuration
    # Each block will launch 16 x 16 = 256 threads
    # 16 threads handle train samples (x-direction), 16 handle test samples (y-direction)
    threads_per_block = (16, 16)

    # Calculate how many blocks are needed to fully cover the data in both dimensions

    # For train samples:
    # We need enough blocks so that (blocks_x * 16) >= n_train
    # The formula (n_train + 15) // 16 does ceiling division
    # Example: if n_train = 100 → (100 + 15) // 16 = 115 // 16 = 7. So 7 blocks in x-direction
    blocks_x = (n_train + threads_per_block[0] - 1) // threads_per_block[0]

    # Same thing for test samples (y-direction)
    blocks_y = (n_test + threads_per_block[1] - 1) // threads_per_block[1]

    # Now we pack it into the 2D grid shape
    blocks_per_grid = (blocks_x, blocks_y)


    # Launch the kernel on the GPU
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

    # Copy distance results back to CPU
    dist_out_host = np.empty((n_test * n_train), dtype=np.float32)
    cuda.memcpy_dtoh(dist_out_host, dist_out_gpu)

    # Reshape into distance matrix (n_test x n_train)
    dist_matrix = dist_out_host.reshape((n_test, n_train))

    # Predict labels using KNN majority voting
    y_pred = []
    for i in range(n_test):
        top_k_idx = np.argsort(dist_matrix[i])[:k]         # indices of k smallest distances
        top_k_labels = y_train[top_k_idx]                  # grab corresponding labels
        vote = Counter(top_k_labels).most_common(1)[0][0]  # majority vote
        y_pred.append(vote)

    return np.array(y_pred)




# Predict
y_pred = np.array(predict_all(k = 11))

# Results
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
    