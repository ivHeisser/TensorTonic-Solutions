#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Write code here
    const int idx = threadIdx.x + blockIdx.x * blockDim.x; // 0..N
    const int idy = threadIdx.y + blockIdx.y * blockDim.y; // 0..M

    C[idy * N + idx] = 0.f;
    
    if ((idx < N) && (idy < M))
        for (int k = 0; k < K; k++)
           C[idy * N + idx] += A[idy * K + k] * B [k * N + idx];
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matmul_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
    cudaGetLastError();
}
