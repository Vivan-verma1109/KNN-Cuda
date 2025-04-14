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