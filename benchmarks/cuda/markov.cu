#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void setup_kernel(float* d_t_matrix, int n_states) {
    
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n_states) return;

    curandState state;
    curand_init(clock64() + idx, 0, 0, &state);

    float sum = 0.0f;

    for (int next_state = 0; next_state < n_states; ++next_state) {
        float value = curand_uniform(&state);
        d_t_matrix[idx*n_states + next_state] = value;
        sum += value;
    }

    for (int next_state = 0; next_state < n_states; ++next_state) {
        d_t_matrix[idx*n_states + next_state] /= sum;
    }

}

__global__ void markov(float* d_t_matrix, float* d_state_vector, float* d_next_state_vector, int n_states) {
    
    int state = blockIdx.x*blockDim.x + threadIdx.x;
    if (state >= n_states) return;

    float sum = 0.0f;
    for (int prev_state = 0; prev_state < n_states; ++prev_state) {
        sum += d_t_matrix[prev_state*n_states + state]*d_state_vector[prev_state];
    }
    d_next_state_vector[state] = sum;
    
}

int main(int argc, char** argv) {
    cudaError_t nb_error;
    cudaEvent_t start, stop;   

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);  

    int n_states = atoi(argv[1]);
    int n_steps = atoi(argv[2]);
    int s_blocks = 128;
    int n_blocks = (n_states + s_blocks - 1) / s_blocks;

    float *d_t_matrix, *d_state_vector, *d_next_state_vector;
    float time;

    // 
    cudaMalloc(&d_t_matrix, n_states*n_states*sizeof(float));
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 1: %s\n", cudaGetErrorString(nb_error));
    // 
    // 
    cudaMalloc(&d_state_vector, n_states*sizeof(float));
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 2: %s\n", cudaGetErrorString(nb_error));
    // 
    // 
    cudaMalloc(&d_next_state_vector, n_states*sizeof(float));
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 3: %s\n", cudaGetErrorString(nb_error));
    // 

    setup_kernel<<<n_blocks, s_blocks>>>(d_t_matrix, n_states);
    cudaDeviceSynchronize();

    float* h_vector = (float*)calloc(n_states, sizeof(float));
    h_vector[0] = 1.0f;
    cudaMemcpy(d_state_vector, h_vector, n_states*sizeof(float), cudaMemcpyHostToDevice);
    free(h_vector);

    for (int i = 0; i < n_steps; ++i) {
        markov<<<n_blocks, s_blocks>>>(d_t_matrix, d_state_vector, d_next_state_vector, n_states);
        cudaDeviceSynchronize();

        float* tmp = d_state_vector;
        d_state_vector = d_next_state_vector;
        d_next_state_vector = tmp;
    }

    float* resultadoFinal = (float*)malloc(n_states*sizeof(float));
    cudaMemcpy(resultadoFinal, d_state_vector, n_states*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Resultado final:\n");
    for (int i = 0; i < n_states; ++i) {
        printf("%f ", resultadoFinal[i]);
    }
    printf("\n");
    
    cudaEventRecord(stop, 0) ;
    cudaEventSynchronize(stop) ;
    cudaEventElapsedTime(&time, start, stop) ;

    printf("tempo: %3.1fms\n", time);
    
    cudaFree(d_t_matrix);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 3: %s\n", cudaGetErrorString(nb_error));

    cudaFree(d_state_vector);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 3: %s\n", cudaGetErrorString(nb_error));

    cudaFree(d_next_state_vector);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 3: %s\n", cudaGetErrorString(nb_error));
    
    cudaEventDestroy(start);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 6: %s\n", cudaGetErrorString(nb_error));
    
    cudaEventDestroy(stop);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 7: %s\n", cudaGetErrorString(nb_error));
    
    free(resultadoFinal);

    return 0;
}
