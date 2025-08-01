#include <stdio.h>
#include <curand_kernel.h>

__global__ void gen_matrix(float* A, int r) {
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    if (idx >= r*r) return;

    curandState state;
    curand_init(clock64(), idx, 0, &state);

    A[idx] = curand_uniform(&state) * 10.0f;
}

// Kernel para um passo da decomposição LU
__global__ void lu_kernel(float* A, float* L, float* U, int r, int i) {
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    if (idx >= r) return;

    // Diagonal de L
    if (idx == i) {
        L[i*r+i] = 1.0f;
    }

    // Linha i de U
    if (idx >= i) {
        float soma = 0.0f;
        for (int j=0; j<i; j++) {
            soma += L[i*r+j]*U[j*r+idx];
        }
        U[i*r+idx] = A[i*r+idx]-soma;
    }

    // Coluna i de L
    if (idx > i) {
        float soma = 0.0f;
        for (int j=0; j<i; j++) {
            soma += L[idx*r+j]*U[j*r+i];
        }
        L[idx*r+i] = (A[idx*r+i]-soma)/U[i*r+i];
    }
}

void print_matrix(float* data, int r) {
    for (int i=0; i<r; i++) {
        for (int j=0; j<r; j++) {
            printf("%8.4f ", data[i*r+j]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    cudaError_t nb_error;
    cudaEvent_t start, stop;   

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int r = atoi(argv[1]);
    size_t tamanho = r*r*sizeof(float);

    float *d_matrix, *d_l, *d_u;
    float *A, *L, *U;
    float time;

    // 
    cudaMalloc(&d_matrix, tamanho);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 1: %s\n", cudaGetErrorString(nb_error));
    // 
    //
    cudaMalloc(&d_l, tamanho);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 2: %s\n", cudaGetErrorString(nb_error));
    // 
    // 
    cudaMalloc(&d_u, tamanho);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 3: %s\n", cudaGetErrorString(nb_error));
    // 
    // Inicialização da matriz A com números aleatórios
    gen_matrix<<<(r*r+255)/256, 256>>>(d_matrix, r);
    cudaDeviceSynchronize();

    // Decomposição LU paralela (passo a passo)
    for (int i = 0; i < r; i++) {
        lu_kernel<<<(r+255)/256, 256>>>(d_matrix, d_l, d_u, r, i);
        cudaDeviceSynchronize();
    }
    
    // Alocação no host e cópia dos resultados
    A = (float*)malloc(tamanho);
    L = (float*)malloc(tamanho);
    U = (float*)malloc(tamanho);

    cudaMemcpy(A, d_matrix, tamanho, cudaMemcpyDeviceToHost);
    cudaMemcpy(L, d_l, tamanho, cudaMemcpyDeviceToHost);
    cudaMemcpy(U, d_u, tamanho, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop, 0) ;
    cudaEventSynchronize(stop) ;
    cudaEventElapsedTime(&time, start, stop) ;
    
    printf("tempo: %3.1fms\n", time);
    // printf("matriz A (original):\n");
    // print_matrix(A, r);
    // printf("matriz L (inferior):\n");
    // print_matrix(L, r);
    // printf("matriz U (superior):\n");
    // print_matrix(U, r);

    // Libera memória
    free(A);
    free(L);
    free(U);

    cudaFree(d_matrix); 
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 4: %s\n", cudaGetErrorString(nb_error));

    cudaFree(d_l); 
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 5: %s\n", cudaGetErrorString(nb_error));

    cudaFree(d_u);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 6: %s\n", cudaGetErrorString(nb_error));

    cudaEventDestroy(start);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 7: %s\n", cudaGetErrorString(nb_error));

    cudaEventDestroy(stop);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 8: %s\n", cudaGetErrorString(nb_error));

    return 0;
}
