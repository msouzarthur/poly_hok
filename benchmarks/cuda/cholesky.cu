#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void cholesky_diag(float *A, int k, int n) {
    __shared__ float sum[256];
    int tid = threadIdx.x;
    float temp = 0;

    for (int j=tid; j<k; j+=blockDim.x) {
        float val = A[k*n+j];
        temp += val*val;
    }
    sum[tid] = temp;
    __syncthreads();

    for (int stride=blockDim.x/2; stride>0; stride/=2) {
        if (tid<stride) sum[tid] += sum[tid+stride];
        __syncthreads();
    }

    if (tid==0) {
        A[k*n+k] = sqrtf(A[k*n+k] - sum[0]);
    }
}

__global__ void cholesky_update_col(float *A, int k, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x+k+1;
    if (i<n) {
        float temp = 0;
        for (int j=0; j<k; j++) {
            temp += A[i*n+j]*A[k*n+j];
        }
        A[i*n+k] = (A[i*n+k]-temp)/A[k*n+k];
    }
}

int main(int argc, char *argv[]) {
    cudaEvent_t start, stop;   
    cudaError_t nb_error;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int n_samples = atoi(argv[1]);
    int s_blocks = 128;
    // int n_blocks = (n_samples+s_blocks-1)/s_blocks;

    float time;
    float *A_host;
    float *A_dev;
    
    A_host = (float *)malloc(n_samples*n_samples*sizeof(float));

    for (int i=0; i<n_samples; i++) {
        for (int j=0; j<n_samples; j++) {
            if (i==j)
                A_host[i*n_samples+j] = 4.0f;
            else
                A_host[i*n_samples+j] = 1.0f;
        }
    }

    for (int i=0; i<n_samples; i++) {
        for (int j=0; j<n_samples; j++) {
            printf("%f\t", A_host[i*n_samples+j]);
        }
        printf("\n");
    }
    //
    cudaMalloc(&A_dev, n_samples*n_samples*sizeof(float));
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 1: %s\n", cudaGetErrorString(nb_error));
    //
    cudaMemcpy(A_dev, A_host, n_samples * n_samples * sizeof(float), cudaMemcpyHostToDevice);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 2: %s\n", cudaGetErrorString(nb_error));
    //

    for (int k = 0; k < n_samples; k++) {
        cholesky_diag<<<1, s_blocks>>>(A_dev, k, n_samples);
        cudaDeviceSynchronize();

        int remaining = n_samples - (k + 1);
        int n_blocks_update = (remaining + s_blocks - 1) / s_blocks;
        if (n_blocks_update > 0) {
            cholesky_update_col<<<n_blocks_update, s_blocks>>>(A_dev, k, n_samples);
            cudaDeviceSynchronize();
        }
    }

    //
    cudaMemcpy(A_host, A_dev, n_samples*n_samples*sizeof(float), cudaMemcpyDeviceToHost);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 2: %s\n", cudaGetErrorString(nb_error));
    //

    cudaEventRecord(stop, 0) ;
    cudaEventSynchronize(stop) ;
    cudaEventElapsedTime(&time, start, stop) ;
    
    printf("resultado:\n");

    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_samples; j++) {
            printf("%f\t", A_host[i*n_samples+j]);  
        }
        printf("\n");
    }
    //
    cudaFree(A_dev);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 4: %s\n", cudaGetErrorString(nb_error));
    //
    cudaEventDestroy(start);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 3: %s\n", cudaGetErrorString(nb_error));
    //
    cudaEventDestroy(stop);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 4: %s\n", cudaGetErrorString(nb_error));
    //

    free(A_host);
    
    return 0;
}
