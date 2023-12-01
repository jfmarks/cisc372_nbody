#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include "compute.h"


__global__ void computeAccels(double* d_hPos, double* d_mass, vector3* d_accels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < NUMENTITIES && j < NUMENTITIES) {
        if (i == j) {
            FILL_VECTOR(d_accels[i*NUMENTITIES + j], 0, 0, 0);
        }
        else {
            vector3 distance;
            for (int k = 0; k < 3; k++) {
                distance[k] = d_hPos[i*3 + k] - d_hPos[j*3 + k];
            }
            double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
            double magnitude = sqrt(magnitude_sq);
            double accelmag = -1 * GRAV_CONSTANT * d_mass[j] / magnitude_sq;
            FILL_VECTOR(d_accels[i*NUMENTITIES + j], accelmag*distance[0] / magnitude, accelmag*distance[1] / magnitude, accelmag*distance[2] / magnitude);
        }
    }
}

__global__ void updateVelPos(vector3* d_hVel, vector3* d_hPos, vector3* d_accels) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < NUMENTITIES) {

        vector3 sum = {0,0,0};
        for (int j = 0; j < NUMENTITIES; j++) {
            for (int k = 0; k < 3; k++) {
                sum[k] += d_accels[i*NUMENTITIES + j][k];
            }
        }
        for (int k = 0; k < 3; k++) {
            d_hVel[i][k] += sum[k] * INTERVAL;
            d_hPos[i][k] += d_hVel[i][k] * INTERVAL;
        }
    }
}

void compute() {
    //vector3 *d_hVel;
    double *d_mass, *d_hPos;
    vector3 *d_accels;

    cudaMalloc((void**)&d_hPos, sizeof(double) * 3 * NUMENTITIES);
    //cudaMalloc((void**)&d_hVel, sizeof(vector3) * NUMENTITIES);
    cudaMalloc((void**)&d_mass, sizeof(double) * NUMENTITIES);
    cudaMalloc((void**)&d_accels, sizeof(vector3) * NUMENTITIES * NUMENTITIES);


    cudaMemcpy(d_hPos, hPos, sizeof(double) * 3 * NUMENTITIES, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_hVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);

    dim3 blockSize(16,16);
    dim3 gridSize((NUMENTITIES + blockSize.x -1)/ blockSize.x, (NUMENTITIES + blockSize.y - 1) / blockSize.y);

    computeAccels<<<gridSize, blockSize>>>(d_hPos, d_mass, d_accels);
    cudaDeviceSynchronize();
    

    vector3* h_accels = (vector3*)malloc(sizeof(vector3) * NUMENTITIES * NUMENTITIES);
    cudaMemcpy(h_accels, d_accels, NUMENTITIES * NUMENTITIES * sizeof(vector3), cudaMemcpyDeviceToHost);

    cudaFree(d_hPos);
    //cudaFree(d_hVel);
    cudaFree(d_mass);
    cudaFree(d_accels);

        // Perform calculations using the computed accelerations
    for (int i = 0; i < NUMENTITIES; i++) {
        vector3 sum = { 0, 0, 0 };
        for (int j = 0; j < NUMENTITIES; j++) {
            for (int k = 0; k < 3; k++) {
                sum[k] += h_accels[i * NUMENTITIES + j][k];
            }
        }

        for (int k = 0; k < 3; k++) {
            hVel[i][k] += sum[k] * INTERVAL;
            hPos[i][k] += hVel[i][k] * INTERVAL;
        }
    }

    free(h_accels);
}
