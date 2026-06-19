#include <cuda_runtime.h>
#include <math.h>

/*
 *     expf() has more accuracy.
 *   __expf() is faster but less accurate (it uses hardware approximation).
*/
__global__ void sigmoid_kernel(const float* input, float* output, int N) {
    // Write code here
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < N; i += stride){
        output[i] = 1.f / (1.f + __expf(-input[i]));
    }
        
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    sigmoid_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}