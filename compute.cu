#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include "compute.h"

#define BLOCK_SIZE 16
#define EPSILON 1.0e-8

__global__ void computeAccels(vector3* d_hPos, double* d_mass, vector3* d_accels) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < NUMENTITIES && j < NUMENTITIES) {
        if (i != j) {
            double distanceX = d_hPos[i][0] - d_hPos[j][0];
            double distanceY = d_hPos[i][1] - d_hPos[j][1];
            double distanceZ = d_hPos[i][2] - d_hPos[j][2];

            double magnitude_sq = distanceX * distanceX + distanceY * distanceY + distanceZ * distanceZ;
            double magnitude = sqrt(magnitude_sq);

            double force = GRAV_CONSTANT * d_mass[i] * d_mass[j] / (magnitude_sq + EPSILON);

            double accX = force * distanceX / magnitude;
            double accY = force * distanceY / magnitude;
            double accZ = force * distanceZ / magnitude;

            d_accels[i * NUMENTITIES + j][0] = accX;
            d_accels[i * NUMENTITIES + j][1] = accY;
            d_accels[i * NUMENTITIES + j][2] = accZ;

        } else {

            d_accels[i * NUMENTITIES + j][0] = 0.0;
            d_accels[i * NUMENTITIES + j][1] = 0.0;
            d_accels[i * NUMENTITIES + j][2] = 0.0;
        }
    }
}

__global__ void updateVelPos(vector3* d_hVel, vector3* d_hPos, vector3* d_accels, double* d_mass) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < NUMENTITIES) {

        double accX = 0.0;
        double accY = 0.0;
        double accZ = 0.0;

        for (int j = 0; j < NUMENTITIES; ++j) {
            accX += d_accels[i * NUMENTITIES + j][0];
            accY += d_accels[i * NUMENTITIES + j][1];
            accZ += d_accels[i * NUMENTITIES + j][2];
        }

        d_hVel[i][0] += accX * INTERVAL;
        d_hVel[i][1] += accY * INTERVAL;
        d_hVel[i][2] += accZ * INTERVAL;

        d_hPos[i][0] += d_hVel[i][0] * INTERVAL;
        d_hPos[i][1] += d_hVel[i][1] * INTERVAL;
        d_hPos[i][2] += d_hVel[i][2] * INTERVAL;
    }
}

void compute() {
    vector3 *d_hPos, *d_hVel;
    double *d_mass;
    vector3 *d_accels;

    cudaMalloc((void**)&d_hPos, sizeof(vector3) * NUMENTITIES);
    cudaMalloc((void**)&d_hVel, sizeof(vector3) * NUMENTITIES);
    cudaMalloc((void**)&d_mass, sizeof(double) * NUMENTITIES);
    cudaMalloc((void**)&d_accels, sizeof(vector3) * NUMENTITIES * NUMENTITIES);

    cudaMemcpy(d_hPos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((NUMENTITIES + blockSize.x - 1) / blockSize.x, (NUMENTITIES + blockSize.y - 1) / blockSize.y);
    computeAccels<<<gridSize, blockSize>>>(d_hPos, d_mass, d_accels);
    cudaDeviceSynchronize();

    updateVelPos<<<gridSize, blockSize>>>(d_hVel, d_hPos, d_accels, d_mass);
    cudaDeviceSynchronize();

    cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, d_hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);

    cudaFree(d_hPos);
    cudaFree(d_hVel);
    cudaFree(d_mass);
    cudaFree(d_accels);
}