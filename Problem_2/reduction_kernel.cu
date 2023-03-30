#define BLOCK_SIZE 1024

__global__ void reduction_kernel(unsigned int *out_hash, unsigned int *out_nonce, const unsigned int *in_hash, const unsigned int *in_nonce, unsigned int size) {
    int index = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ unsigned int reduction_hash[BLOCK_SIZE];
    __shared__ unsigned int reduction_nonce[BLOCK_SIZE];

    reduction_hash[threadIdx.x] = UINT_MAX;
    reduction_nonce[threadIdx.x] = 0;

    if (index < size) {
        reduction_hash[threadIdx.x] = in_hash[index];
        reduction_nonce[threadIdx.x] = in_nonce[index];
    }

    if ((index + BLOCK_SIZE) < size) {
        if (in_hash[index + BLOCK_SIZE] < reduction_hash[threadIdx.x]) {
            reduction_hash[threadIdx.x] = in_hash[index + BLOCK_SIZE];
            reduction_nonce[threadIdx.x] = in_nonce[index + BLOCK_SIZE];
        }
    }

    for (int stride = BLOCK_SIZE / 2; stride >= 1; stride = stride / 2) {
        __syncthreads();
        if (threadIdx.x < stride) {
            if (reduction_hash[threadIdx.x + stride] < reduction_hash[threadIdx.x]) {
                reduction_hash[threadIdx.x] = reduction_hash[threadIdx.x + stride];
                reduction_nonce[threadIdx.x] = reduction_nonce[threadIdx.x + stride];
            }
        }
    }

    if (threadIdx.x == 0) {
        out_hash[blockIdx.x] = reduction_hash[0];
        out_nonce[blockIdx.x] = reduction_nonce[0];
    }
}