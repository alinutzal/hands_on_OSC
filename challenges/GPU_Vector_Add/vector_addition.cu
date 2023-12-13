#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

int main() {
    const int n = 1000000; // Size of the vectors
    float *x, *y, *d_x, *d_y;
    float alpha = 1.0;

    // Allocate host memory
    x = (float*)malloc(n * sizeof(float));
    y = (float*)malloc(n * sizeof(float));

    // Initialize host arrays
    for (int i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i);
        y[i] = static_cast<float>(i);
    }

    // Allocate device memory
    cudaMalloc(&d_x, n * sizeof(float)); 
    cudaMalloc(&d_y, n * sizeof(float));

    // Copy host memory to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform the vector addition: y = alpha * x + y
    cublasSaxpy(handle, n, &alpha, d_x, 1, d_y, 1);

    // Copy result back to host
    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    int N = 1; 
    std::cout << "\n---------------------------\n";
    std::cout << "__SUCCESS__\n";
    std::cout << "---------------------------\n";
    std::cout << "N                 = %d\n" <<  N;
    //printf("Threads Per Block = %d\n", thr_per_blk);
    //printf("Blocks In Grid    = %d\n", blk_in_grid);
    std::cout << "---------------------------\n\n";


    // Clean up resources
    cudaFree(d_x);
    cudaFree(d_y);
    cublasDestroy(handle);
    free(x);
    free(y);

    return 0;
}