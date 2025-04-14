#include <cuda_runtime.h> 

__global__ void compute_distances(float *x_train, float *x_test, float *dist_out, int n_train, int D){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int test_idx  = blockIdx.y * blockDim.y + threadIdx.y;
    int train_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    
    if (idx >= n_train) return;
    
    float dist = 0;
    for(int d = 0; d < D; d++){
        float diff = x_train[train_idx * D + d] - x_test[test_idx * D + d]
        dist += diff * diff
    }
    dist_out[test_idx * N + train_idx] = dist;
}