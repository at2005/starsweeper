#include <cuda_runtime.h>
#include <stdio.h>


__global__ leverage_kernel(double* trajectory, double3* observed_positions, double3* predicted_positions, double3*int N) {
    double alpha = 0;
    double beta = 0;
    for(int i = 0; i < N; i++) {
        double t = trajectory[i];
        alpha += (t*t); 
        beta += t;
    }

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double t_i = trajectory[idx];
    double h_ii = alpha*alpha - 2 * t_i * beta + N * t_i * t_i;
    h_ii = h_ii / (N * alpha - (beta * beta));

    double3 residuals = make_double3(
        observed_positions[idx].x - predicted_positions[idx].x,
        observed_positions[idx].y - predicted_positions[idx].y,
        observed_positions[idx].z - predicted_positions[idx].z
    );

}