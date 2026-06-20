#include <cuda_runtime.h>
#include <math.h>
#include <cstdio>

__global__ void gelu_kernel(const float* input, float* output, int N) {
    // Write code here
    const float inv_sqrt_2 = 0.7071067811865475f; // 1/sqrt(2)

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < (unsigned int)N; i += stride) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + erff(x * inv_sqrt_2));
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    dim3 blocks((N + 255) / 256);
    gelu_kernel<<<blocks, threads>>>(input, output, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}
