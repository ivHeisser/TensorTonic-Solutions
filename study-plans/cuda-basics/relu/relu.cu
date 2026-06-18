#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    // Write code here
    register const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    register const int stride = blockDim.x * gridDim.x;
    // grid-stride loop
    for (register int i = idx; i < N; i+=stride) {
        output[i] = fmaxf(input[i], 0.0f);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}