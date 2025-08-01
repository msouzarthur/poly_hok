#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void dot_product_kernel(float* A, float* Q, float* R, int k, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i <= k || i >= n) return;

    float dot = 0.0f;
    for (int j=0; j<n; j++) {
        dot += Q[j*n+k]*A[j*n+i];
    }
    R[k*n+i] = dot;
}

__global__ void update_column_kernel(float* A, float* Q, float* R, int k, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int col = k+1+i;
    if (col >= n) return;

    for (int row=0; row<n; row++) {
        A[row*n+col] -= R[k*n+col]*Q[row*n+k];
    }
}

__global__ void normalize_column_kernel(float* A, float* Q, float* R, int k, int n) {
    __shared__ float sum[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x*blockDim.x+tid;

    float val = 0;
    if (idx < n) {
        val = A[idx*n+k];
        sum[tid] = val*val;
    } else {
        sum[tid] = 0.0f;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) sum[tid] += sum[tid + stride];
        __syncthreads();
    }

    float norm = sqrtf(sum[0]);
    if (norm < 1e-10f) norm = 1.0f; // para evitar divisão por zero

    if (idx < n) {
        Q[idx*n+k] = A[idx*n+k] / norm;
    }
    if (tid == 0) {
        R[k*n+k] = norm;
    }
}

int main(int argc, char *argv[]) {
    cudaEvent_t start, stop;   
    cudaError_t nb_error;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);  

    int n_samples = atoi(argv[1]);
    
    float time;
    float *A_host;
    float *A_dev;
    float *Q_dev;
    float *R_dev;

    A_host = (float *)malloc(n_samples*n_samples*sizeof(float));

    for (int i=0; i<n_samples*n_samples; i++) {
        A_host[i] = (float)(rand()%100)/100.0f;
    }

    cudaMalloc(&A_dev, n_samples*n_samples*sizeof(float));
    cudaMalloc(&Q_dev, n_samples*n_samples*sizeof(float));
    cudaMalloc(&R_dev, n_samples*n_samples*sizeof(float));

    cudaMemcpy(A_dev, A_host, n_samples*n_samples*sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n_samples+threads-1)/threads;

    for (int k=0; k<n_samples; k++) {
        normalize_column_kernel<<<blocks, threads>>>(A_dev, Q_dev, R_dev, k, n_samples);
        cudaDeviceSynchronize();

        dot_product_kernel<<<blocks, threads>>>(A_dev, Q_dev, R_dev, k, n_samples);
        cudaDeviceSynchronize();

        update_column_kernel<<<blocks, threads>>>(A_dev, Q_dev, R_dev, k, n_samples);
        cudaDeviceSynchronize();
    }

    //
    cudaMemcpy(A_host, Q_dev, n_samples*n_samples*sizeof(float), cudaMemcpyDeviceToHost);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 3: %s\n", cudaGetErrorString(nb_error));
    // 
    printf("Q[0,0] = %f\n", A_host[0]);

    cudaEventRecord(stop, 0) ;
    cudaEventSynchronize(stop) ;
    cudaEventElapsedTime(&time, start, stop) ;
    //
    cudaFree(A_dev);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 4: %s\n", cudaGetErrorString(nb_error));
    //
    cudaFree(Q_dev);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 4: %s\n", cudaGetErrorString(nb_error));
    //
    cudaFree(R_dev);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 4: %s\n", cudaGetErrorString(nb_error));
    //
    cudaEventDestroy(start);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 6: %s\n", cudaGetErrorString(nb_error));
    //
    cudaEventDestroy(stop);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 7: %s\n", cudaGetErrorString(nb_error));
    //
    free(A_host);

    return 0;
}
