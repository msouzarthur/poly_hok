#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

void checkCuda(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s: %s\n", message, cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

__global__ void calculateReturns(const float* prices, float* returns, int numAssets, int numDays) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int totalReturns = numAssets * (numDays - 1);
    if (index >= totalReturns) return;

    int assetIndex = index / (numDays - 1);
    int dayIndex = index % (numDays - 1);

    float priceToday = prices[assetIndex * numDays + dayIndex];
    float priceTomorrow = prices[assetIndex * numDays + dayIndex + 1];

    returns[index] = (priceTomorrow - priceToday) / priceToday;
}

__global__ void calculateSMA(const float* prices, float* sma, int numAssets, int numDays, int windowSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = numAssets * (numDays - windowSize + 1);
    if (index >= total) return;

    int assetIndex = index / (numDays - windowSize + 1);
    int dayIndex = index % (numDays - windowSize + 1);

    float sum = 0.0f;
    for (int i = 0; i < windowSize; ++i) {
        sum += prices[assetIndex * numDays + dayIndex + i];
    }
    sma[index] = sum / windowSize;
}

__global__ void calculateVariance(const float* returns, float* variance, int numAssets, int numDays, int windowSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = numAssets * (numDays - windowSize + 1);
    if (index >= total) return;

    int assetIndex = index / (numDays - windowSize + 1);
    int dayIndex = index % (numDays - windowSize + 1);

    float mean = 0.0f;
    for (int i = 0; i < windowSize; ++i) {
        mean += returns[assetIndex * (numDays - 1) + dayIndex + i];
    }
    mean /= windowSize;

    float varSum = 0.0f;
    for (int i = 0; i < windowSize; ++i) {
        float deviation = returns[assetIndex * (numDays - 1) + dayIndex + i] - mean;
        varSum += deviation * deviation;
    }
    variance[index] = varSum / windowSize;
}

__global__ void calculateMomentum(const float* prices, float* momentum, int numAssets, int numDays, int period) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = numAssets * (numDays - period);
    if (index >= total) return;

    int assetIndex = index / (numDays - period);
    int dayIndex = index % (numDays - period);

    momentum[index] = prices[assetIndex * numDays + dayIndex + period] - prices[assetIndex * numDays + dayIndex];
}

__global__ void calculateSmaCrossCount(const float* prices, const float* sma, int* counts, int numAssets, int numDays, int windowSize) {
    int assetIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (assetIndex >= numAssets) return;

    int counter = 0;
    for (int i = 0; i < numDays - windowSize + 1; ++i) {
        if (prices[assetIndex * numDays + i + windowSize - 1] > sma[assetIndex * (numDays - windowSize + 1) + i]) {
            counter++;
        }
    }
    counts[assetIndex] = counter;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Usage: %s <num_assets> <num_days> <window_size>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int numAssets = atoi(argv[1]);
    int numDays = atoi(argv[2]);
    int windowSize = atoi(argv[3]);
    int period = windowSize;

    if (numDays < windowSize) {
        fprintf(stderr, "Error: num_days must be >= window_size\n");
        return EXIT_FAILURE;
    }

    size_t pricesSize = numAssets * numDays * sizeof(float);
    size_t returnsSize = numAssets * (numDays - 1) * sizeof(float);
    size_t windowOutputSize = numAssets * (numDays - windowSize + 1) * sizeof(float);
    size_t momentumSize = numAssets * (numDays - period) * sizeof(float);
    size_t countSize = numAssets * sizeof(int);

    float* h_prices = (float*)malloc(pricesSize);
    float* h_returns = (float*)malloc(returnsSize);
    float* h_sma = (float*)malloc(windowOutputSize);
    float* h_variance = (float*)malloc(windowOutputSize);
    float* h_momentum = (float*)malloc(momentumSize);
    int* h_counts = (int*)malloc(countSize);

    for (int i = 0; i < numAssets; ++i) {
        for (int j = 0; j < numDays; ++j) {
            h_prices[i * numDays + j] = 100.0f + j * 0.1f + (float)(rand() % 100) / 1000.0f;
        }
    }

    float *d_prices, *d_returns, *d_sma, *d_variance, *d_momentum;
    int* d_counts;

    checkCuda(cudaMalloc(&d_prices, pricesSize), "alloc d_prices");
    checkCuda(cudaMalloc(&d_returns, returnsSize), "alloc d_returns");
    checkCuda(cudaMalloc(&d_sma, windowOutputSize), "alloc d_sma");
    checkCuda(cudaMalloc(&d_variance, windowOutputSize), "alloc d_variance");
    checkCuda(cudaMalloc(&d_momentum, momentumSize), "alloc d_momentum");
    checkCuda(cudaMalloc(&d_counts, countSize), "alloc d_counts");

    checkCuda(cudaMemcpy(d_prices, h_prices, pricesSize, cudaMemcpyHostToDevice), "copy h_prices to d_prices");

    int threadsPerBlock = 256;
    int blocksReturns = (numAssets * (numDays - 1) + threadsPerBlock - 1) / threadsPerBlock;
    int blocksWindow = (numAssets * (numDays - windowSize + 1) + threadsPerBlock - 1) / threadsPerBlock;
    int blocksMomentum = (numAssets * (numDays - period) + threadsPerBlock - 1) / threadsPerBlock;
    int blocksCount = (numAssets + threadsPerBlock - 1) / threadsPerBlock;

    calculateReturns<<<blocksReturns, threadsPerBlock>>>(d_prices, d_returns, numAssets, numDays);
    checkCuda(cudaDeviceSynchronize(), "calculateReturns");

    calculateSMA<<<blocksWindow, threadsPerBlock>>>(d_prices, d_sma, numAssets, numDays, windowSize);
    checkCuda(cudaDeviceSynchronize(), "calculateSMA");

    calculateVariance<<<blocksWindow, threadsPerBlock>>>(d_returns, d_variance, numAssets, numDays, windowSize);
    checkCuda(cudaDeviceSynchronize(), "calculateVariance");

    calculateMomentum<<<blocksMomentum, threadsPerBlock>>>(d_prices, d_momentum, numAssets, numDays, period);
    checkCuda(cudaDeviceSynchronize(), "calculateMomentum");

    calculateSmaCrossCount<<<blocksCount, threadsPerBlock>>>(d_prices, d_sma, d_counts, numAssets, numDays, windowSize);
    checkCuda(cudaDeviceSynchronize(), "calculateSmaCrossCount");

    checkCuda(cudaMemcpy(h_returns, d_returns, returnsSize, cudaMemcpyDeviceToHost), "copy d_returns to h_returns");
    checkCuda(cudaMemcpy(h_sma, d_sma, windowOutputSize, cudaMemcpyDeviceToHost), "copy d_sma to h_sma");
    checkCuda(cudaMemcpy(h_variance, d_variance, windowOutputSize, cudaMemcpyDeviceToHost), "copy d_variance to h_variance");
    checkCuda(cudaMemcpy(h_momentum, d_momentum, momentumSize, cudaMemcpyDeviceToHost), "copy d_momentum to h_momentum");
    checkCuda(cudaMemcpy(h_counts, d_counts, countSize, cudaMemcpyDeviceToHost), "copy d_counts to h_counts");

    printf("Asset 0 - Sample Results:\n");
    printf("Returns: ");
    for (int i = 0; i < 5 && i < numDays - 1; ++i) printf("%.6f ", h_returns[i]);
    printf("\nSMA: ");
    for (int i = 0; i < 5 && i < numDays - windowSize + 1; ++i) printf("%.6f ", h_sma[i]);
    printf("\nVariance: ");
    for (int i = 0; i < 5 && i < numDays - windowSize + 1; ++i) printf("%.6f ", h_variance[i]);
    printf("\nMomentum: ");
    for (int i = 0; i < 5 && i < numDays - period; ++i) printf("%.6f ", h_momentum[i]);
    printf("\nPrice > SMA days: %d\n", h_counts[0]);

    cudaFree(d_prices);
    cudaFree(d_returns);
    cudaFree(d_sma);
    cudaFree(d_variance);
    cudaFree(d_momentum);
    cudaFree(d_counts);

    free(h_prices);
    free(h_returns);
    free(h_sma);
    free(h_variance);
    free(h_momentum);
    free(h_counts);

    return EXIT_SUCCESS;
}
//
/*TESTAR
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

void checkCuda(cudaError_t ret, const char* msg) {
    if (ret != cudaSuccess) {
        fprintf(stderr, "CUDA Error %s: %s\n", msg, cudaGetErrorString(ret));
        exit(1);
    }
}

__global__ void calc_returns(const float* prices, float* returns, int numAssets, int numDays) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = numAssets * (numDays - 1);
    if (idx >= total) return;

    int asset = idx / (numDays - 1);
    int day = idx % (numDays - 1);

    float p1 = prices[asset * numDays + day];
    float p2 = prices[asset * numDays + day + 1];

    returns[idx] = (p2 - p1) / p1;
}

__global__ void calc_sma(const float* prices, float* sma, int numAssets, int numDays, int windowSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = numAssets * (numDays - windowSize + 1);
    if (idx >= total) return;

    int asset = idx / (numDays - windowSize + 1);
    int day = idx % (numDays - windowSize + 1);

    float sum = 0.0f;
    for (int i = 0; i < windowSize; i++) {
        sum += prices[asset * numDays + day + i];
    }
    sma[idx] = sum / windowSize;
}

__global__ void calc_variance(const float* prices, const float* sma, float* variance, int numAssets, int numDays, int windowSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = numAssets * (numDays - windowSize + 1);
    if (idx >= total) return;

    int asset = idx / (numDays - windowSize + 1);
    int day = idx % (numDays - windowSize + 1);

    float mean = sma[idx];
    float sum_sq_diff = 0.0f;
    for (int i = 0; i < windowSize; i++) {
        float diff = prices[asset * numDays + day + i] - mean;
        sum_sq_diff += diff * diff;
    }
    variance[idx] = sum_sq_diff / windowSize;
}

__global__ void calc_stddev_from_variance(const float* variance, float* stddev, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    stddev[idx] = sqrtf(variance[idx]);
}

__global__ void calc_bollinger_bands(const float* sma, const float* stddev, float* upper_band, float* lower_band,
                                     int numAssets, int numDays, int windowSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = numAssets * (numDays - windowSize + 1);
    if (idx >= total) return;

    upper_band[idx] = sma[idx] + 2.0f * stddev[idx];
    lower_band[idx] = sma[idx] - 2.0f * stddev[idx];
}

__global__ void calc_momentum(const float* prices, float* momentum, int numAssets, int numDays, int momentumWindow) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = numAssets * (numDays - momentumWindow);
    if (idx >= total) return;

    int asset = idx / (numDays - momentumWindow);
    int day = idx % (numDays - momentumWindow);

    momentum[idx] = prices[asset * numDays + day + momentumWindow] - prices[asset * numDays + day];
}

__global__ void calc_counts(const float* prices, const float* sma, int* counts, int numAssets, int numDays, int windowSize) {
    int asset = blockIdx.x * blockDim.x + threadIdx.x;
    if (asset >= numAssets) return;

    int count = 0;
    for (int day = 0; day < numDays - windowSize + 1; day++) {
        if (prices[asset * numDays + day + windowSize - 1] > sma[asset * (numDays - windowSize + 1) + day]) {
            count++;
        }
    }
    counts[asset] = count;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Uso: %s <numAssets> <numDays> <windowSize>\n", argv[0]);
        return 1;
    }

    int numAssets = atoi(argv[1]);
    int numDays = atoi(argv[2]);
    int windowSize = atoi(argv[3]);
    int momentumWindow = windowSize;

    if (numDays < windowSize) {
        printf("Erro: numDays deve ser >= windowSize\n");
        return 1;
    }

    printf("numAssets = %d, numDays = %d, windowSize = %d\n", numAssets, numDays, windowSize);

    float* h_prices = (float*)malloc(numAssets * numDays * sizeof(float));
    float* h_returns = (float*)malloc(numAssets * (numDays - 1) * sizeof(float));
    float* h_sma = (float*)malloc(numAssets * (numDays - windowSize + 1) * sizeof(float));
    float* h_variance = (float*)malloc(numAssets * (numDays - windowSize + 1) * sizeof(float));
    float* h_stddev = (float*)malloc(numAssets * (numDays - windowSize + 1) * sizeof(float));
    float* h_upper_band = (float*)malloc(numAssets * (numDays - windowSize + 1) * sizeof(float));
    float* h_lower_band = (float*)malloc(numAssets * (numDays - windowSize + 1) * sizeof(float));
    float* h_momentum = (float*)malloc(numAssets * (numDays - momentumWindow) * sizeof(float));
    int* h_counts = (int*)malloc(numAssets * sizeof(int));

    // Inicializa preços simulados
    for (int i = 0; i < numAssets; i++) {
        for (int j = 0; j < numDays; j++) {
            h_prices[i * numDays + j] = 100.0f + j * 0.1f + (float)(rand() % 100) / 1000.0f;
        }
    }

    float *d_prices, *d_returns, *d_sma, *d_variance, *d_stddev, *d_upper_band, *d_lower_band, *d_momentum;
    int* d_counts;

    checkCuda(cudaMalloc(&d_prices, numAssets * numDays * sizeof(float)), "cudaMalloc prices");
    checkCuda(cudaMalloc(&d_returns, numAssets * (numDays - 1) * sizeof(float)), "cudaMalloc returns");
    checkCuda(cudaMalloc(&d_sma, numAssets * (numDays - windowSize + 1) * sizeof(float)), "cudaMalloc sma");
    checkCuda(cudaMalloc(&d_variance, numAssets * (numDays - windowSize + 1) * sizeof(float)), "cudaMalloc variance");
    checkCuda(cudaMalloc(&d_stddev, numAssets * (numDays - windowSize + 1) * sizeof(float)), "cudaMalloc stddev");
    checkCuda(cudaMalloc(&d_upper_band, numAssets * (numDays - windowSize + 1) * sizeof(float)), "cudaMalloc upper_band");
    checkCuda(cudaMalloc(&d_lower_band, numAssets * (numDays - windowSize + 1) * sizeof(float)), "cudaMalloc lower_band");
    checkCuda(cudaMalloc(&d_momentum, numAssets * (numDays - momentumWindow) * sizeof(float)), "cudaMalloc momentum");
    checkCuda(cudaMalloc(&d_counts, numAssets * sizeof(int)), "cudaMalloc counts");

    checkCuda(cudaMemcpy(d_prices, h_prices, numAssets * numDays * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy prices");

    int threads = 256;
    int blocks_returns = (numAssets * (numDays - 1) + threads - 1) / threads;
    int blocks_sma = (numAssets * (numDays - windowSize + 1) + threads - 1) / threads;
    int blocks_variance = blocks_sma;
    int blocks_stddev = blocks_sma;
    int blocks_bollinger = blocks_sma;
    int blocks_momentum = (numAssets * (numDays - momentumWindow) + threads - 1) / threads;
    int blocks_counts = (numAssets + threads - 1) / threads;

    calc_returns<<<blocks_returns, threads>>>(d_prices, d_returns, numAssets, numDays);
    checkCuda(cudaDeviceSynchronize(), "calc_returns");

    calc_sma<<<blocks_sma, threads>>>(d_prices, d_sma, numAssets, numDays, windowSize);
    checkCuda(cudaDeviceSynchronize(), "calc_sma");

    calc_variance<<<blocks_variance, threads>>>(d_prices, d_sma, d_variance, numAssets, numDays, windowSize);
    checkCuda(cudaDeviceSynchronize(), "calc_variance");

    calc_stddev_from_variance<<<blocks_stddev, threads>>>(d_variance, d_stddev, numAssets * (numDays - windowSize + 1));
    checkCuda(cudaDeviceSynchronize(), "calc_stddev_from_variance");

    calc_bollinger_bands<<<blocks_bollinger, threads>>>(d_sma, d_stddev, d_upper_band, d_lower_band, numAssets, numDays, windowSize);
    checkCuda(cudaDeviceSynchronize(), "calc_bollinger_bands");

    calc_momentum<<<blocks_momentum, threads>>>(d_prices, d_momentum, numAssets, numDays, momentumWindow);
    checkCuda(cudaDeviceSynchronize(), "calc_momentum");

    calc_counts<<<blocks_counts, threads>>>(d_prices, d_sma, d_counts, numAssets, numDays, windowSize);
    checkCuda(cudaDeviceSynchronize(), "calc_counts");

    checkCuda(cudaMemcpy(h_returns, d_returns, numAssets * (numDays - 1) * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy returns");
    checkCuda(cudaMemcpy(h_sma, d_sma, numAssets * (numDays - windowSize + 1) * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy sma");
    checkCuda(cudaMemcpy(h_variance, d_variance, numAssets * (numDays - windowSize + 1) * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy variance");
    checkCuda(cudaMemcpy(h_stddev, d_stddev, numAssets * (numDays - windowSize + 1) * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy stddev");
    checkCuda(cudaMemcpy(h_upper_band, d_upper_band, numAssets * (numDays - windowSize + 1) * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy upper_band");
    checkCuda(cudaMemcpy(h_lower_band, d_lower_band, numAssets * (numDays - windowSize + 1) * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy lower_band");
    checkCuda(cudaMemcpy(h_momentum, d_momentum, numAssets * (numDays - momentumWindow) * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy momentum");
    checkCuda(cudaMemcpy(h_counts, d_counts, numAssets * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy counts");

    printf("Ativo 0 - primeiros 5 retornos:\n");
    for (int i = 0; i < 5 && i < numDays - 1; i++) printf("%.6f ", h_returns[i]);
    printf("\n");

    printf("Ativo 0 - primeiros 5 SMA:\n");
    for (int i = 0; i < 5 && i < numDays - windowSize + 1; i++) printf("%.6f ", h_sma[i]);
    printf("\n");

    printf("Ativo 0 - primeiros 5 variâncias:\n");
    for (int i = 0; i < 5 && i < numDays - windowSize + 1; i++) printf("%.9f ", h_variance[i]);
    printf("\n");

    printf("Ativo 0 - primeiros 5 desvios padrão:\n");
    for (int i = 0; i < 5 && i < numDays - windowSize + 1; i++) printf("%.6f ", h_stddev[i]);
    printf("\n");

    printf("Ativo 0 - primeiros 5 bandas superiores:\n");
    for (int i = 0; i < 5 && i < numDays - windowSize + 1; i++) printf("%.6f ", h_upper_band[i]);
    printf("\n");

    printf("Ativo 0 - primeiros 5 bandas inferiores:\n");
    for (int i = 0; i < 5 && i < numDays - windowSize + 1; i++) printf("%.6f ", h_lower_band[i]);
    printf("\n");

    printf("Ativo 0 - primeiros 5 momentum:\n");
    for (int i = 0; i < 5 && i < numDays - momentumWindow; i++) printf("%.6f ", h_momentum[i]);
    printf("\n");

    printf("Ativo 0 - dias com preço > SMA: %d\n", h_counts[0]);

    cudaFree(d_prices);
    cudaFree(d_returns);
    cudaFree(d_sma);
    cudaFree(d_variance);
    cudaFree(d_stddev);
    cudaFree(d_upper_band);
    cudaFree(d_lower_band);
    cudaFree(d_momentum);
    cudaFree(d_counts);

    free(h_prices);
    free(h_returns);
    free(h_sma);
    free(h_variance);
    free(h_stddev);
    free(h_upper_band);
    free(h_lower_band);
    free(h_momentum);
    free(h_counts);

    return 0;
}
*/