
// this is kinda half baked rn, gonna test it out in python for sedna first

#define MAX_HEIGHT 240
#define MAX_WIDTH 240
#define RA_CENTER_IDX 120
#define DEC_CENTER_IDX 120
#define DEG_PER_PIXEL 0.00006944444

struct OrbitalElements {
    double ascending_node;
    double inclination;
};

struct ICRS_Coords {
    double ra;
    double dec;
}

__global__ void pathfinder(uint8_t* image, ICRS_Coords* coords, double3 reference_earth_pos, double3 current_earth_pos, OrbitalElements* elements) {
    uint32_t* indices = new uint32_t[MAX_HEIGHT * MAX_WIDTH];
    int i = threadIdx.y;
    int j = threadIdx.x;
    double  cos_o, cos_i, sin_i, cos_dec_current, sin_dec_current, cos_ra_current, sin_ra_current;
    double ra_center = coords[blockIdx.x].ra;
    double dec_center = coords[blockIdx.y].dec;
    double dec_current = dec_center + (i - DEC_CENTER_IDX) * DEG_PER_PIXEL;
    double ra_current = ra_center + (j - RA_CENTER_IDX) * DEG_PER_PIXEL;
    double dec_current_rad = radians(dec_current);
    double ra_current_rad = radians(ra_current);
    sincos(dec_current_rad, &sin_dec_current, &cos_dec_current);
    sincos(ra_current_rad, &sin_ra_current, &cos_ra_current);
    sincos(elements->inclination, &sin_i, &cos_i);
    cos_o = cos(elements->ascending_node);
    double x = cos_dec_current * cos_ra_current;
    double y = cos_dec_current * sin_ra_current;
    double z = sin_dec_current;
    double3 parallax = make_double3(
        current_earth_pos.x - reference_earth_pos.x,
        current_earth_pos.y - reference_earth_pos.y,
        current_earth_pos.z - reference_earth_pos.z
    );
    
    // account for axial tilt and adjust for parallax, then move to heliocentric frame
    x -= parallax.x - current_earth_pos.x; 
    y = fma(0.91747714,y, fma(0.39778851, z, - parallax.y - current_earth_pos.y));
    z = fma(-0.39778851,y, fma(0.91747714, z, - parallax.z - current_earth_pos.z));

    // we only care about the last row of the rotation matrix
    double r_20 = sin(elements->inclination + elements->ascending_node);
    double r_21 = sin_i * cos_o;
    double r_22 = cos_i;
   
    double z_p = r_20 * x + r_21 * y + r_22 * z;
    double is_in_ellipse = (double)(abs(z_p) < 1e-6);
    int idx = i * MAX_WIDTH + j;
    
    // image[idx] = fma(is_in_ellipse, 255, (uint8_t)((1.00 - is_in_ellipse) * image[idx]));
}


// stacks along the path of a single window ie PS cutout
__global__ void stack_along_path_singular(uint8_t* idx_array, uint8_t* d3_image) {
    uint8_t idx = idx_array[threadIdx.x];
    

}

// stacks along the path of multiple windows
__global__ void stack_along_windows() {}


int main() {
    ICRS_Coords* coords = new ICRS_Coords[100];
    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 10; j++) {
            coords[i].ra = i * 0.1;
            coords[i].dec = j * 0.1;
        }
    }

    uint8_t* device_image;
    cudaMalloc(&device_image, MAX_HEIGHT * MAX_WIDTH);
    cudaMemset(device_image, 0, MAX_HEIGHT * MAX_WIDTH);
    ICRS_Coords* device_coords;
    cudaMalloc(&device_coords, 100 * sizeof(ICRS_Coords));
    cudaMemcpy(device_coords, coords, 100 * sizeof(ICRS_Coords), cudaMemcpyHostToDevice);

    OrbitalElements elements;
    elements.ascending_node = 0.1;
    elements.inclination = 0.1;
    OrbitalElements* device_elements;
    cudaMalloc(&device_elements, sizeof(OrbitalElements));
    cudaMemcpy(device_elements, &elements, sizeof(OrbitalElements), cudaMemcpyHostToDevice);
    
}