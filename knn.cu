#include <cuda_runtime.h> 
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

__global__ void majority_vote(int *top_k_labels, int *y_pred, int n_test, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_test) return;

    int count = 0
    for(int j = 0; j < k; j++){
         count += top_k_labels[idx * k + j];
    }
    y_pred[idx] = (count >= (k + 1) / 2) ? 1 : 0; //If the number of 1s is greater than or equal to half of k → predict 1, else predict 0
}