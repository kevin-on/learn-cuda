#include <cstdlib>
#include <ctime>
#include <cuda/cmath>
#include <cuda_runtime_api.h>
#include <memory.h>
#include <stdio.h>

__global__ void vecAdd(float *A, float *B, float *C, int vectorLength) {
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIndex < vectorLength) {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

void initArray(float *A, int vectorLength) {
    std::srand(std::time({}));
    for (int i = 0; i < vectorLength; i++) {
        A[i] = rand() / (float)RAND_MAX;
    }
}

void serialVecAdd(float *A, float *B, float *C, int vectorLength) {
    for (int i = 0; i < vectorLength; i++) {
        C[i] = A[i] + B[i];
    }
}

bool vectorApproximatelyEqual(float *A, float *B, int length) {
    for (int i = 0; i < length; i++) {
        if (fabs(A[i] - B[i]) > 1e-5)
            return false;
    }
    return true;
}

void unifiedMemVecAdd(int vectorLength) {
    float *A = nullptr;
    float *B = nullptr;
    float *hostC = (float *)malloc(vectorLength * sizeof(float));
    float *devC = nullptr;

    // Unified memory allocation
    cudaMallocManaged(&A, vectorLength * sizeof(float));
    cudaMallocManaged(&B, vectorLength * sizeof(float));
    cudaMallocManaged(&devC, vectorLength * sizeof(float));

    initArray(A, vectorLength);
    initArray(B, vectorLength);

    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(A, B, devC, vectorLength);

    cudaDeviceSynchronize(); // What happens if we do not synchronize?

    serialVecAdd(A, B, hostC, vectorLength);

    if (vectorApproximatelyEqual(hostC, devC, vectorLength)) {
        printf("Unified Memory: CPU and GPU answers match\n");
    } else {
        printf("Unified Memory: Error - CPU and GPU answers do not match\n");
    }

    // Cleanup
    cudaFree(A);
    cudaFree(B);
    cudaFree(devC);
    free(hostC);
}

void explicitMemVecAdd(int vectorLength) {
    float *A = nullptr;
    float *B = nullptr;
    float *C = (float *)malloc(vectorLength * sizeof(float));
    float *copiedDevC = nullptr;
    float *devA = nullptr;
    float *devB = nullptr;
    float *devC = nullptr;

    cudaMallocHost(&A, vectorLength * sizeof(float));
    cudaMallocHost(&B, vectorLength * sizeof(float));
    cudaMallocHost(&C, vectorLength * sizeof(float));

    cudaMalloc(&devA, vectorLength * sizeof(float));
    cudaMalloc(&devB, vectorLength * sizeof(float));
    cudaMalloc(&devC, vectorLength * sizeof(float));

    initArray(A, vectorLength);
    initArray(B, vectorLength);

    cudaMemcpy(devA, A, vectorLength * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(devB, B, vectorLength * sizeof(float), cudaMemcpyDefault);
    cudaMemset(devC, 0, vectorLength * sizeof(float));

    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);

    cudaDeviceSynchronize();

    serialVecAdd(A, B, C, vectorLength);

    cudaMemcpy(copiedDevC, devC, vectorLength * sizeof(float), cudaMemcpyDefault);

    if (vectorApproximatelyEqual(C, copiedDevC, vectorLength)) {
        printf("Explicit Memory: CPU and GPU answers match\n");
    } else {
        printf("Explicit Memory: Error - CPU and GPU answers do not match\n");
    }

    // Cleanup
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(copiedDevC);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    free(C);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: %s <vectorLength> <mode(1: unified mem, 2: explicit mem)>\n", argv[0]);
        return 1;
    }
    int vectorLength = atoi(argv[1]);
    int mode = atoi(argv[2]);
    if (mode == 1) {
        unifiedMemVecAdd(vectorLength);
    } else if (mode == 2) {
        explicitMemVecAdd(vectorLength);
    } else {
        printf("Invalid mode\n");
        return 1;
    }
}