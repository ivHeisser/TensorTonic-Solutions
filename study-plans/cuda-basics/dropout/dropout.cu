#include <cuda_runtime.h>

__global__ void dropout_kernel(const float* input, const float* mask, float* output, float p, int N) {
    // Write code here
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride)
        output[i] = input[i] * mask[i] / (1.f - p);
}

extern "C" void solve(const float* input, const float* mask, float* output, float p, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    dropout_kernel<<<blocks, threads>>>(input, mask, output, p, N);
    cudaDeviceSynchronize();
}