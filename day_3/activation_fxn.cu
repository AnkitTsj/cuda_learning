#include <stdio.h>
#include <cmath>
#include <cuda_runtime.h>

#define TOTAL_VALS 1000
#define THREADS 256


__global__ void relu(float * array_1,float * array_2,int n){
    int indices = blockDim.x * blockIdx.x + threadIdx.x;
    if(indices < n){
        array_2[indices] = fmax(0.0,array_1[indices]);
    }
}

__global__ void sigmoid(float * array_1,float * array_2,int n){
    int indices = blockDim.x * blockIdx.x + threadIdx.x;
    if(indices<n){
        array_2[indices] = 1.0f/(1.0f+expf(-array_1[indices]));
    }
}

__global__ void softmax(float * array_1,float*array_2,float * global_sum,int n){
    __shared__ float exp_sum[THREADS];
    int indices = blockDim.x * blockIdx.x + threadIdx.x;
    float expn = (indices < n) ? expf(array_1[indices]): 0.0f;
    __syncthreads();
    float exp_sum = expn;
    for(int i = blockDim.x/2;i>0;i/=2){
        if(threadIdx.x < indices){
            exp_sum[threadIdx.x] += exp_sum[threadIdx.x + i];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        atomicAdd(global_sum,exp_sum[0]);
    }
    __syncthreads();
    if(indices < n){
    array_2[indices] = expn/ *global_sum;}


__host__ void check_results(float* cpu_result, float* gpu_result, int n, float tolerance = 1e-6) {
    for (int i = 0; i < n; i++) {
        if (fabs(cpu_result[i] - gpu_result[i]) > tolerance) {
            printf("Mismatch at index %d, CPU  = %d GPU = %d ",i,cpu_result[i],gpu_result[i]);
            break;
        }
    }
    printf("All results match.");
}

    
}

void verifyResults(float *host_x, float *host_y, int n) {
    float sum_exp = 0;
    for (int i = 0; i < n; i++) {
        sum_exp += expf(host_x[i]);
    }

    bool correct = true;
    for (int i = 0; i < n; i++) {
        float expected = expf(host_x[i]) / sum_exp;
        if (fabs(host_y[i] - expected) > 1e-6) {
            printf("Mismatch at index %d,expected %d but got %d"i,expected, host_y[i]);
            correct = false;
            break;
        }
    }
    if (correct) {
        printf("Softmax output verified successfully!!");
    } else {
        printf("Softmax verification failed!");
    }
}
int main(){
    int blocks = (TOTAL_VALS+THREADS-1)/TOTAL_VALS;
    float * host_relu,* host_sigmoid,* host_softmax,*host_x;
    float * device_relu,* device_sigmoid, * device_softmax,*device_x;
    int mem_size = sizeof(float) * TOTAL_VALS;
    host_x = (float *)malloc(mem_size);
    host_relu = (float *)malloc(mem_size);
    host_sigmoid = (float *)malloc(mem_size);
    host_softmax = (float *)malloc(mem_size);
    cudaMalloc(&device_relu,mem_size);
    cudaMalloc(&device_sigmoid,mem_size);
    cudaMalloc(&device_softmax,mem_size);
    for (int i = 0; i < N; i++) {
        host_x[i] = (float)(rand() % 200 - 100) / 10.0f; 
    }
    cudaMemcpy(device_x, host_x, mem_size, cudaMemcpyHostToDevice);
    relu<<<blocks,THREADS>>>(device_x,device_relu,TOTAL_VALS);
    cudaMemcpy(host_relu, device_relu, mem_size, cudaMemcpyDeviceToHost);
    verifyResults(host_x, host_relu, TOTAL_VALS, "ReLU");
    sigmoid<<<blocks,THREADS>>>(device_x,device_sigmoid,TOTAL_VALS);
    cudaMemcpy(host_sigmoid, device_x, N * sizeof(float), cudaMemcpyDeviceToHost);
    check_results(host_x, host_sigmoid, N, "Sigmoid");



    cudaFree(device_x);
    cudaFree(device_relu);
    cudaFree(device_sigmoid);
    cudaFree(device_softmax);
    
    delete[] host_x;
    delete[] host_relu;
    delete[] host_sigmoid;
    delete[] host_softmax;


}
