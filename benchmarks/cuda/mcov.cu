#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void mean_kernel(float *data, float *mean, int n_samples, int n_var) {
    int j = threadIdx.x+blockIdx.x*blockDim.x;
    if (j >= n_var) return;

    float sum = 0.0f;

    for (int k = 0; k < n_samples; k++) {
        sum += data[k*n_var+j];
    }

    mean[j] = sum/n_samples;
}

__global__ void cov_kernel(float *data, float *mean, float *cov, int n_samples, int n_var) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    float sum = 0.0f;
    float a = 0.0f;
    float b = 0.0f;
    
    if (i >= n_var || j >= n_var) return;

    for (int k = 0; k < n_samples; k++) {
        a = data[k*n_var+i]-mean[i];
        b = data[k*n_var+j]-mean[j];
        sum += a*b;
    }

    cov[i*n_var+j] = sum/(n_samples-1);
}

int main(int argc, char *argv[]) {
    cudaEvent_t start, stop;   
    cudaError_t nb_error;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);  

    int n_samples = atoi(argv[1]);
    int n_var = atoi(argv[2]);
    int s_blocks = 128;
    int n_blocks = (n_samples+s_blocks-1)/s_blocks;

    float time;
    float *d_data, *d_mean, *d_cov;

    float *h_data = (float*)malloc(n_samples*n_var*sizeof(float));
    for (int i = 0; i < n_samples*n_var; i++) {
        h_data[i] = (float)(rand() % 100);
    }

    // 
    cudaMalloc(&d_data, n_samples*n_var*sizeof(float));
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 1: %s\n", cudaGetErrorString(nb_error));
    // 
    // 
    cudaMalloc(&d_mean, n_var*sizeof(float));
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 2: %s\n", cudaGetErrorString(nb_error));
    // 
    // 
    cudaMalloc(&d_cov, n_var*n_var*sizeof(float));
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 3: %s\n", cudaGetErrorString(nb_error));
    // 
    
    cudaMemcpy(d_data, h_data, n_samples*n_var*sizeof(float), cudaMemcpyHostToDevice);

    mean_kernel<<<n_blocks, s_blocks>>>(d_data, d_mean, n_samples, n_var);

    cov_kernel<<<n_var, n_var>>>(d_data, d_mean, d_cov, n_samples, n_var);

    float h_cov[n_var*n_var];
    cudaMemcpy(h_cov, d_cov, n_var*n_var*sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop, 0) ;
    cudaEventSynchronize(stop) ;
    cudaEventElapsedTime(&time, start, stop) ;
    
    printf("tempo: %3.1fms\n", time);

    cudaFree(d_data);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 4: %s\n", cudaGetErrorString(nb_error));
    
    cudaFree(d_mean);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 5: %s\n", cudaGetErrorString(nb_error));
    
    cudaFree(d_cov);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 6: %s\n", cudaGetErrorString(nb_error));
        
    cudaEventDestroy(start);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 6: %s\n", cudaGetErrorString(nb_error));
    
    cudaEventDestroy(stop);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 7: %s\n", cudaGetErrorString(nb_error));
    
    free(h_data);

    return 0;
}