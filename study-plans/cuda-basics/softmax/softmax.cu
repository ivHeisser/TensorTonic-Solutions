#include <cuda_runtime.h>
/*
Problem: softmax without max_val is numerically unstable.

The correct way to compute softmax is:

exp(x_i - max_x) /
sum_j exp(x_j - max_x)

1. Perform a reduction to find the maximum value.
2. Perform a reduction to compute the sum of exponentials relative to that maximum.
3. Compute the final result.

Otherwise, for values such as x = 100 or x = 1000,
floating-point overflow can occur very quickly.
*/

__global__ void softmax_kernel(const float* input, float* output, int N) {
    // Write code here
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    float sum_exp = 0.0f;
    float max_val = input[0];
    
    for (unsigned int i = 0; i < N; i++)
        max_val = fmaxf(max_val, input[i]); // TODO: implement parallel max__val__reduce
    for (unsigned int i = 0; i < N; i++)
        sum_exp += __expf(input[i] - max_val); // TODO: implement parallel sum__exp__reduce  
    for (unsigned int i = idx; i < N; i += stride)
        output[i] = __expf(input[i] - max_val) / sum_exp;
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}