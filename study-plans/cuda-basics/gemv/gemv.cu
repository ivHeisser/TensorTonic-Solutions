#include <cuda_runtime.h>

__global__ void gemv_kernel_v2(const float* __restrict__ A, const float* __restrict__ x, float* __restrict__ y, int M, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < M; i += stride) {
        const float* row = A + (size_t)i * N;
        float sum = 0.0f;

        #pragma unroll 4
        for (int j = 0; j < N; ++j)
            sum += row[j] * x[j];

        y[i] = sum;
    }
}

__global__ void gemv_kernel_v1(const float* A, const float* x, float* __restrict__ y, int M, int N) { //optimized version
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;

    const float* row = A + (size_t)i * N;
    float sum = 0.0f;
    int N4 = N / 4;

    const float4* row4 = reinterpret_cast<const float4*>(row);
    const float4* x4   = reinterpret_cast<const float4*>(x);

    for (int j = 0; j < N4; ++j) {
        float4 a = row4[j];
        float4 b = x4[j];

        sum += a.x * b.x;
        sum += a.y * b.y;
        sum += a.z * b.z;
        sum += a.w * b.w;
    }
    
    #pragma unroll 4
    for (int j = N4 * 4; j < N; ++j) { // Tail (in case N is not a multiple of 4)
        sum += row[j] * x[j];
    }

    y[i] = sum;
}


__global__ void gemv_kernel_v0(const float* A, const float* x, float* y, int M, int N) {
    // Write code here
    // y[i]= ∑ A[i,j]⋅x[j], row-major
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    float sum = 0.0f;
    int stride = gridDim.x * blockDim.x;
    
    for(int j = 0; j < N; j++)
        sum += A[i * N + j] * x[j];
    y[i] = sum;    
}


extern "C" void solve(const float* A, const float* x, float* y, int M, int N) {
    dim3 threads(256); // to increase perf : try 128 / 256 / 512 threads
    dim3 blocks((M + 255) / 256);
    gemv_kernel_v0<<<blocks, threads>>>(A, x, y, M, N);
    cudaDeviceSynchronize();
    gemv_kernel_v2<<<blocks, threads>>>(A, x, y, M, N);
    cudaDeviceSynchronize();
}
