#include <cuda_runtime.h>

__global__ void hadamard_kernel(const float* A, const float* B, float* C, int M, int N) {
    // Write code here
    const int idx = threadIdx.x + blockIdx.x * blockDim.x; // 0..N
    const int idy = threadIdx.y + blockIdx.y * blockDim.y; // 0..M

    if ((idx < N) & (idy < M)) {
        const int ij = idy * N + idx;
        C[ij] = A[ij] * B[ij]; 
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    hadamard_kernel<<<blocks, threads>>>(A, B, C, M, N);
    cudaDeviceSynchronize();
}
