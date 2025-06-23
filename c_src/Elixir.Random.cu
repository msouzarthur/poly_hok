#include <curand_kernel.h>

__device__ float atomic_random() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(clock64(), tid, 0, &state);
    return curand_uniform(&state);
}