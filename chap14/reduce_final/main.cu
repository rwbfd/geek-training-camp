#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>

using namespace std::chrono;

template<unsigned int blockSize>
__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template<unsigned int blockSize>
__global__ void reduce_final(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = 0;
    while (i < n) {
        sdata[tid] += g_idata[i] + g_idata[i + blockSize];
        i += gridSize;
    }
    __syncthreads();
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; }
        __syncthreads();
    }
    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


int main(void) {
    int N = 100000000;
    float *g_indata_host, *g_indata_device, *g_outdata_host, *g_outdata_device;
    g_indata_host = (float *) malloc(N * sizeof(float));
    g_outdata_host = (float *) malloc(sizeof(float));

    cudaMalloc(&g_indata_device, N * sizeof(float));
    cudaMalloc(&g_outdata_device, sizeof(float));

    for (auto i = 0; i < N; i++) {
        g_indata_host[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);;
    }

    cudaMemcpy(g_indata_device, g_indata_host, N * sizeof(float), cudaMemcpyHostToDevice);

//    This is where the code is run
    auto dimGrid = 512;
    auto dimBlock = 512;
    auto smemSize = 128 * sizeof(float);
    auto threads = 512;
    auto start = high_resolution_clock::now();
    auto n = 2048;
    switch (threads) {
        case 512:
            reduce_final<512><<<dimGrid, dimBlock, smemSize>>>(g_indata_device, g_outdata_device, n);
            break;
        case 256:
            reduce_final<256><<<dimGrid, dimBlock, smemSize>>>(g_indata_device, g_outdata_device, n);
            break;
        case 128:
            reduce_final<128><<<dimGrid, dimBlock, smemSize>>>(g_indata_device, g_outdata_device, n);
            break;
        case 64:
            reduce_final<64><<<dimGrid, dimBlock, smemSize>>>(g_indata_device, g_outdata_device, n);
            break;
        case 32:
            reduce_final<32><<<dimGrid, dimBlock, smemSize>>>(g_indata_device, g_outdata_device, n);
            break;
        case 16:
            reduce_final<16><<<dimGrid, dimBlock, smemSize>>>(g_indata_device, g_outdata_device, n);
            break;
        case 8:
            reduce_final<8><<<dimGrid, dimBlock, smemSize>>>(g_indata_device, g_outdata_device, n);
            break;
        case 4:
            reduce_final<4><<<dimGrid, dimBlock, smemSize>>>(g_indata_device, g_outdata_device, n);
            break;
        case 2:
            reduce_final<2><<<dimGrid, dimBlock, smemSize>>>(g_indata_device, g_outdata_device, n);
            break;
        case 1:
            reduce_final<1><<<dimGrid, dimBlock, smemSize>>>(g_indata_device, g_outdata_device, n);
            break;
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Time taken by function: "
              << duration.count() << " microseconds" << std::endl;
    cudaFree(g_indata_device);
    cudaFree(g_outdata_device);
    free(g_indata_host);
    free(g_outdata_host);

}