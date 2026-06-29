#include <cuda_runtime.h>
#include <math.h>

__global__ void layer_norm_kernel_v1(
// classical reduction for "mu" and "sigma"
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int M,
    int N,
    float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float buf[256];   // blockDim.x <= 256
    __shared__ float mean;
    __shared__ float var;

    // -------- mean --------

    float sum = 0.0f;

    for (int i = tid; i < N; i += blockDim.x)
        sum += input[row * N + i];

    buf[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            buf[tid] += buf[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        mean = buf[0] / N;

    __syncthreads();

    // -------- variance --------

    float sq = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        float d = input[row * N + i] - mean;
        sq += d * d;
    }

    buf[tid] = sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            buf[tid] += buf[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        var = buf[0] / N;

    __syncthreads();

    // -------- normalize --------

    float inv_std = rsqrtf(var + eps);

    for (int i = tid; i < N; i += blockDim.x) {
        float x = input[row * N + i];
        output[row * N + i] =
            (x - mean) * inv_std * gamma[i] + beta[i];
    }
}


__global__ void layer_norm_kernel(
//"mean" и "var" evaluate only the "threadIdx.x == 0"
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int M,
    int N,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= M) return;

    __shared__ float mean;
    __shared__ float var;

    // Calculate mean и variance by the same one thread
    if (tid == 0)
    {
        float sum = 0.0f;

        for (int j = 0; j < N; j++)
            sum += input[row * N + j];

        mean = sum / N;

        float sq_sum = 0.0f;

        for (int j = 0; j < N; j++)
        {
            float d = input[row * N + j] - mean;
            sq_sum += d * d;
        }

        var = sq_sum / N;
    }

    __syncthreads();

    float inv_std = rsqrtf(var + eps);

    // Each thread processes its own portion of the columns
    for (int col = tid; col < N; col += blockDim.x)
    {
        float x = input[row * N + col];
        output[row * N + col] =
            (x - mean) * inv_std * gamma[col] + beta[col];
    }
}


extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M);
    layer_norm_kernel<<<blocks, threads>>>(input, gamma, beta, output, M, N, eps);
    cudaDeviceSynchronize();
}
