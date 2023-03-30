#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <driver_types.h>
#include <curand.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <cstdio>
#include <cuda.h>

#include "support.h"
#include "hash_kernel.cu"
#include "nonce_kernel.cu"
#include "reduction_kernel.cu"

// to activate debug statements
#define DEBUG 1

// program constants
#define BLOCK_SIZE 1024
#define SEED       123

// solution constants
#define MAX     123123123
#define TARGET  20

//CITATIONS:
//chat GPT

// functions used
unsigned int generate_hash(unsigned int nonce, unsigned int index, unsigned int* transactions, unsigned int n_transactions);
void read_file(char* file, unsigned int* transactions, unsigned int n_transactions);
void err_check(cudaError_t ret, char* msg, int exit_code);


/* Main ------------------ //
*   This is the main program.
*/
int main(int argc, char* argv[]) {

    // Catch console errors
    if (argc != 6) {
        printf("USE LIKE THIS: gpu_mining transactions.csv n_transactions trials out.csv time.csv\n");
        return EXIT_FAILURE;
    }


    // Output files
    FILE* output_file = fopen(argv[4], "w");
    FILE* time_file   = fopen(argv[5], "w");

    // Read in the transactions
    unsigned int n_transactions = strtoul(argv[2], NULL, 10);
    unsigned int* transactions = (unsigned int*)calloc(n_transactions, sizeof(unsigned int));
    read_file(argv[1], transactions, n_transactions);

    // get the number of trials
    unsigned int trials = strtoul(argv[3], NULL, 10);


    // -------- Start Mining ------------------------------------------------------- //
    // ----------------------------------------------------------------------------- //
    
    // Set timer and cuda error return
    Timer timer;
    startTime(&timer);
    cudaError_t cuda_ret;

    // To use with kernels
    int num_blocks = ceil((float)trials / (float)BLOCK_SIZE);
    dim3 dimGrid(num_blocks, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);


    // ------ Step 1: generate the nonce values ------ //

    // Allocate the nonce device memory
    unsigned int* device_nonce_array;
    cuda_ret = cudaMalloc((void**)&device_nonce_array, trials * sizeof(unsigned int));
    err_check(cuda_ret, (char*)"Unable to allocate nonces to device memory!", 1);

    // Launch the nonce kernel
    nonce_kernel <<< dimGrid, dimBlock >>> (
        device_nonce_array, // put nonces into here
        trials,             // size of array
        MAX,                // to mod with
        SEED                // random seed
        );
    cuda_ret = cudaDeviceSynchronize();
    err_check(cuda_ret, (char*)"Unable to launch nonce kernel!", 2);

    // Get nonces from device memory
    unsigned int* nonce_array = (unsigned int*)calloc(trials, sizeof(unsigned int));
    cuda_ret = cudaMemcpy(nonce_array, device_nonce_array, trials * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    err_check(cuda_ret, (char*)"Unable to read nonce from device memory!", 3);


    // ------ Step 2: Generate the hash values ------ //

    // TODO Problem 1: perform this hash generation in the GPU
    // Hint: You need both nonces and transactions to compute a hash.
    unsigned int* device_hash_array;
    cuda_ret = cudaMalloc((void**)&device_hash_array, trials * sizeof(unsigned int));
    err_check(cuda_ret, (char*)"Unable to allocate hash array to device memory!", 4);

    // Allocate transactions array in device memory
    unsigned int* device_transactions;
    cuda_ret = cudaMalloc((void**)&device_transactions, n_transactions * sizeof(unsigned int));
    cudaMemcpy(device_transactions, transactions, n_transactions* sizeof(unsigned int), cudaMemcpyHostToDevice);


    // Launch the hash kernel
    hash_kernel <<< dimGrid, dimBlock >>> (
        device_hash_array,  // put hashes into here
        device_nonce_array, // use these nonces
        trials,             // size of arrays
        device_transactions,       // transactions to use in the hash
        n_transactions,     // number of transactions
        MAX                 // to mod with
        );

    cuda_ret = cudaDeviceSynchronize();
    err_check(cuda_ret, (char*)"Unable to launch hash kernel!", 5);

    // Get the hash array from device memory
    unsigned int* hash_array = (unsigned int*)calloc(trials, sizeof(unsigned int));
    cuda_ret = cudaMemcpy(hash_array, device_hash_array, trials * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    err_check(cuda_ret, (char*)"Unable to read hash from device memory!", 6);

    free(nonce_array);
    free(hash_array);
    free(transactions);
    cudaFree(device_transactions);
    

    // ------ Step 3: Find the nonce with the minimum hash value ------ //
    // TODO Problem 2: find the minimum in the GPU by reduction
    
   // Allocate device memory for the minimum hash values and nonces
   unsigned int* d_min_hash;
   unsigned int* d_min_nonce;
   cudaMalloc(&d_min_hash, (trials / (2 * BLOCK_SIZE)) * sizeof(unsigned int));
   cudaMalloc(&d_min_nonce, (trials / (2 * BLOCK_SIZE)) * sizeof(unsigned int));

   
   // Set the block and grid dimensions for the reduction kernel
   dim3 dimBlock2(BLOCK_SIZE, 1, 1);
   dim3 dimGrid2((trials / (2 * BLOCK_SIZE)), 1, 1);

   // Launch the reduction kernel to find the local min hash values and local min nonce values
   reduction_kernel<<<dimGrid2, dimBlock2>>>(d_min_hash, d_min_nonce, device_hash_array, device_nonce_array, trials);

   // Copy the local min hash values and local min nonce values to the system memory
   unsigned int min_hash = MAX;
   unsigned int min_nonce = MAX;
   unsigned int* minHashArray = (unsigned int*)malloc((trials / (2 * BLOCK_SIZE)) * sizeof(unsigned int));
   unsigned int* minNonceArray = (unsigned int*)malloc((trials / (2 * BLOCK_SIZE)) * sizeof(unsigned int));
   cudaMemcpy(minHashArray, d_min_hash, (trials / (2 * BLOCK_SIZE)) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
   cudaMemcpy(minNonceArray, d_min_nonce, (trials / (2 * BLOCK_SIZE)) * sizeof(unsigned int), cudaMemcpyDeviceToHost);

   // Find the global min hash value and min nonce value serially using the CPU

   for (int i = 0; i < (trials / (2 * BLOCK_SIZE)); i++) {
       if (minHashArray[i] < min_hash) {
           min_hash = minHashArray[i];
           min_nonce = minNonceArray[i];
       }
   }

    
    // Free memory
    cudaFree(d_min_hash);
    cudaFree(d_min_nonce);
    cudaFree(device_hash_array);
    cudaFree(device_nonce_array);
    free(minHashArray);
    free(minNonceArray);


    stopTime(&timer);
    // ----------------------------------------------------------------------------- //
    // -------- Finish Mining ------------------------------------------------------ //


    // Get if suceeded
    char* res = (char*)malloc(8 * sizeof(char));
    if (min_hash < TARGET)  res = (char*)"Success!";
    else                    res = (char*)"Failure.";

    // Show results in console
    if (DEBUG) 
        printf("%s\n   Min hash:  %u\n   Min nonce: %u\n   %f seconds\n",
            res,
            min_hash,
            min_nonce,
            elapsedTime(timer)
        );

    // Print results
    fprintf(output_file, "%s\n%u\n%u\n", res, min_hash, min_nonce);
    fprintf(time_file, "%f\n", elapsedTime(timer));

    // Cleanup
    fclose(time_file);
    fclose(output_file);

    return 0;
} // End Main -------------------------------------------- //



/* Generate Hash ----------------------------------------- //
*   Generates a hash value from a nonce and transaction list.
*/
unsigned int generate_hash(unsigned int nonce, unsigned int index, unsigned int* transactions, unsigned int n_transactions) {

    unsigned int hash = (nonce + transactions[0] * (index + 1)) % MAX;
    for (int j = 1; j < n_transactions; j++) {
        hash = (hash + transactions[j] * (index + 1)) % MAX;
    }
    return hash;

} // End Generate Hash ---------- //



/* Read File -------------------- //
*   Reads in a file of transactions. 
*/
void read_file(char* file, unsigned int* transactions, unsigned int n_transactions) {

    // open file
    FILE* trans_file = fopen(file, "r");
    if (trans_file == NULL)
        fprintf(stderr, "ERROR: could not read the transaction file.\n"),
        exit(-1);

    // read items
    char line[100] = { 0 };
    for (int i = 0; i < n_transactions && fgets(line, 100, trans_file); ++i) {
        char* p;
        transactions[i] = strtof(line, &p);
    }

    fclose(trans_file);

} // End Read File ------------- //



/* Error Check ----------------- //
*   Exits if there is a CUDA error.
*/
void err_check(cudaError_t ret, char* msg, int exit_code) {
    if (ret != cudaSuccess)
        fprintf(stderr, "%s \"%s\".\n", msg, cudaGetErrorString(ret)),
        exit(exit_code);
} // End Error Check ----------- //
