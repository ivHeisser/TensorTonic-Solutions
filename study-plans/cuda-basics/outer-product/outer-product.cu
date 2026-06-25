#include <cuda_runtime.h>

__global__ void outer_product_kernel(const float* a, const float* b, float* C, const int M, const int N) {
    // Write code here
    const int idx = threadIdx.x + blockIdx.x * blockDim.x; // 0..N
    const int idy = threadIdx.y + blockIdx.y * blockDim.y; // 0..M

    if ((idx < N) && ( idy < M)) 
        C[idy * N + idx] = a[idy] * b[idx];
}

extern "C" void solve(const float* a, const float* b, float* C, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    outer_product_kernel<<<blocks, threads>>>(a, b, C, M, N);
    cudaDeviceSynchronize();
}
