#include <cuda_runtime.h>
#include <stdio.h>

__device__ double3 diff(double3 pos1, double3 pos2) {
    double3 diff;
    diff.x = pos1.x - pos2.x;
    diff.y = pos1.y - pos2.y; 
    diff.z = pos1.z - pos2.z;
    return diff;
}

// appends pos2 to pos1 in-place
__device__ void append(double3& pos1, double3 pos2) {
    pos1.x +=  pos2.x;
    pos1.y +=  pos2.y;
    pos1.z +=  pos2.z;
}

__device__ double dot(double3 a, double3 b) {
    return fma(a.x * b.x, fma(a.y, b.y, fma(a.z, b.z)));
}

__device__ double norm(double3 a) {
    return sqrt(dot(a, a));
}


// calculates the angular deflection of p9
__device__ double calculate_angle(double impact_parameter, double M) {
    double G = 6.67408e-11;
    double c = 299792458;
    return (G * M * 4) / (c*c * impact_parameter);
}

__device__ double calculate_threshold(double3) {
    
}

// given a set of N datapoints for the trajectory of planet nine, 
// calculate angular deflection
__device__ void deflection_calculator(double3* trajectory, int N) {
    if(N < 2) return;
    double3 avg = diff(trajectory[0], trajectory[1]);
    for(int i = 1; i < N; i++) {
        double3 pos = trajectory[i];
        double3 prev = trajectory[i-1];
        double3 r = diff(pos, prev);
        // is this in alignment with the average?
        double alignment = dot(avg, r) / (norm(avg) * norm(r));
        double angle = acos(alignment);
        double threshold = 0.1;
        if(angle < threshold) {

        } else {
            append(avg, r);
        }
    }
}