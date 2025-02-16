#include <stdio.h>
#include <time.h>





int main(int argc, char *argv[])
{

    float *a, *b, *resp;
	float *dev_a, *dev_b, *dev_resp;
    cudaError_t j_error;

    int N = atoi(argv[1]);

    a = (float*)malloc(N*sizeof(float));
    b = (float*)malloc(N*sizeof(float));
    resp = (float*)malloc(N*sizeof(float));

    for(int i=0; i<N; i++) {
		a[i] = rand();
		
	}

    for(int i=0; i<N; i++) {
		b[i] = rand();
		
	}

    int threadsPerBlock = 256;
    int  numberOfBlocks = (N + threadsPerBlock - 1)/ threadsPerBlock;

    float time;
    cudaEvent_t start, stop;   
    cudaEventCreate(&start) ;
    cudaEventCreate(&stop) ;
    cudaEventRecord(start, 0) ;


	cudaMalloc((void**)&dev_a, N*sizeof(float));
    j_error = cudaGetLastError();
    if(j_error != cudaSuccess) {printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1);}
	cudaMalloc((void**)&dev_b, N*sizeof(float));
    j_error = cudaGetLastError();
    if(j_error != cudaSuccess) {printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1);}
	cudaMalloc((void**)&dev_resp, N*sizeof(float));
    j_error = cudaGetLastError();
    if(j_error != cudaSuccess) {printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1);}
	cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
     j_error = cudaGetLastError();
    if(j_error != cudaSuccess) {printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1);}
    cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice);
     j_error = cudaGetLastError();
    if(j_error != cudaSuccess) {printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1);}

    float (*f1)(float,float) = (float (*)(float,float)) get_anonymous_mult_ptr();
    float (*f2)(float,float) = (float (*)(float,float)) get_anonymous_sum_ptr();

    float *final, *d_final;
    final = (float *)malloc(sizeof(float));
	cudaMalloc((void **) &d_final,sizeof(float));

    map_2kernel<<< numberOfBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_resp, N, f1);
     j_error = cudaGetLastError();
    if(j_error != cudaSuccess) {printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1);}

    reduce_kernel<<< numberOfBlocks, threadsPerBlock>>>(dev_resp, d_final, f2, N);
    j_error = cudaGetLastError();
    if(j_error != cudaSuccess) {printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1);}

     cudaMemcpy( final, d_final, sizeof(float), cudaMemcpyDeviceToHost );
     j_error = cudaGetLastError();
    if(j_error != cudaSuccess) {printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1);}

    
    cudaFree(dev_a);
	cudaFree(dev_b);
    cudaFree(dev_resp);
    cudaFree(d_final);
    
	cudaEventRecord(stop, 0) ;
    cudaEventSynchronize(stop) ;
    cudaEventElapsedTime(&time, start, stop) ;

    printf("CUDA\t%d\t%3.1f\n", N,time);

/*
    for(int i=0; i<10; i++) {
		printf("resp[%d] = %f;\n",i,resp[i]);
	}

*/
	//printf("\n FINAL RESULTADO: %f \n", c);

	free(a);
    free(b);
	free(resp);
    free(final);

}
