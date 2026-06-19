#include <cuda_runtime.h>
#include <math.h>

/*
 *     expf() has more accuracy.
 *   __expf() is faster but less accurate (it uses hardware approximation).
*/
__global__ void tanh_kernel(const float* input, float* output, int N) {
    // Write code here
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        float x = input[i];
        output[i] = (__expf(x) - __expf(-x)) / (__expf(x) + __expf(-x));
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    tanh_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}