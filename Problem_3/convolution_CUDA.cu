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
#include "kernel.cu"


#define BLOCK_SIZE 32
#define BILLION  1000000000.0
#define MAX_LINE_LENGTH 25000

//CITATIONS:
//Lecture slides, class exercises

int main (int argc, char *argv[])
{
    // Check console errors
    if( argc != 6)
    {
        printf("USE LIKE THIS: convolution_serial n_row n_col mat_input.csv mat_output.csv time.csv\n");
        return EXIT_FAILURE;
    }

    // Get dims
    int n_row = strtol(argv[1], NULL, 10);
    int n_col = strtol(argv[2], NULL, 10);

    // Get files to read/write 
    FILE* inputFile1 = fopen(argv[3], "r");
    if (inputFile1 == NULL){
        printf("Could not open file %s",argv[3]);
        return EXIT_FAILURE;
    }
    FILE* outputFile = fopen(argv[4], "w");
    FILE* timeFile  = fopen(argv[5], "w");

    // Matrices to use
    int* filterMatrix_h = (int*)malloc(5 * 5 * sizeof(int));
    int* inputMatrix_h  = (int*) malloc(n_row * n_col * sizeof(int));
    int* outputMatrix_h = (int*) malloc(n_row * n_col * sizeof(int));

    // read the data from the file
    int row_count = 0;
    char line[MAX_LINE_LENGTH] = {0};
    while (fgets(line, MAX_LINE_LENGTH, inputFile1)) {
        if (line[strlen(line) - 1] != '\n') printf("\n");
        char *token;
        const char s[2] = ",";
        token = strtok(line, s);
        int i_col = 0;
        while (token != NULL) {
            inputMatrix_h[row_count*n_col + i_col] = strtol(token, NULL,10 );
            i_col++;
            token = strtok (NULL, s);
        }
        row_count++;
    }


    // Filling filter
	// 1 0 0 0 1 
	// 0 1 0 1 0 
	// 0 0 1 0 0 
	// 0 1 0 1 0 
	// 1 0 0 0 1 
    for(int i = 0; i< 5; i++)
        for(int j = 0; j< 5; j++)
            filterMatrix_h[i*5+j]=0;

    filterMatrix_h[0*5+0] = 1;
    filterMatrix_h[1*5+1] = 1;
    filterMatrix_h[2*5+2] = 1;
    filterMatrix_h[3*5+3] = 1;
    filterMatrix_h[4*5+4] = 1;
    
    filterMatrix_h[4*5+0] = 1;
    filterMatrix_h[3*5+1] = 1;
    filterMatrix_h[1*5+3] = 1;
    filterMatrix_h[0*5+4] = 1;

    fclose(inputFile1); 

    //definte the dimensions
    dim3 dimGrid(ceil(n_col/BLOCK_SIZE), ceil(n_row/BLOCK_SIZE), 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    struct timespec start, end;   
    clock_gettime(CLOCK_REALTIME, &start);

    int* inputMatrix_d;
    cudaMalloc((void**)&inputMatrix_d, (n_row * n_col * sizeof(int)));
    cudaMemcpy(inputMatrix_d, inputMatrix_h, (n_row * n_col * sizeof(int)), cudaMemcpyHostToDevice);
 
    int* outputMatrix_d;
    cudaMalloc((void**)&outputMatrix_d, (n_row * n_col * sizeof(int)));


    // Allocate transactions array in device memory
    int* filterMatrix_d;
    cudaMalloc((void**)&filterMatrix_d, (5*5*sizeof(int)));
    cudaMemcpy(filterMatrix_d, filterMatrix_h, (5*5*sizeof(int)), cudaMemcpyHostToDevice);
 

    clock_gettime(CLOCK_REALTIME, &end);
    double time_spent_HostDevice = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / BILLION;

    // --------------------------------------------------------------------------- //
    // ------ Algorithm Start ---------------------------------------------------- //
    clock_gettime(CLOCK_REALTIME, &start);

	// Launch the hash kernel
    kernel <<< dimGrid, dimBlock >>> (
        inputMatrix_d,  // put hashes into here
        outputMatrix_d, // use these nonces
        filterMatrix_d,             // size of arrays
        n_row,       // transactions to use in the hash
        n_col     // number of transactions
        );

    cudaDeviceSynchronize();
    


    clock_gettime(CLOCK_REALTIME, &end);
    double time_spent_LaunchKernel = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / BILLION;
    // --------------------------------------------------------------------------- //
    // ------ Algorithm End ------------------------------------------------------ //

    clock_gettime(CLOCK_REALTIME, &start);

    cudaMemcpy(outputMatrix_h, outputMatrix_d, (n_row*n_col*sizeof(int)), cudaMemcpyDeviceToHost);

    clock_gettime(CLOCK_REALTIME, &end);
    double time_spent_DeviceHost = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / BILLION;


	// Save output matrix as csv file
    for (int i = 0; i<n_row; i++)
    {
        for (int j = 0; j<n_col; j++)
        {
            fprintf(outputFile, "%d", outputMatrix_h[i*n_col +j]);
            if (j != n_col -1)
                fprintf(outputFile, ",");
            else if ( i < n_row-1)
                fprintf(outputFile, "\n");
        }
    }

    // Print time
    fprintf(timeFile, "%.20f\n%.20f\n%.20f", time_spent_HostDevice, time_spent_LaunchKernel, time_spent_DeviceHost);

    // Cleanup
    fclose (outputFile);
    fclose (timeFile);

    free(inputMatrix_h);
    free(outputMatrix_h);
    free(filterMatrix_h);
    cudaFree(inputMatrix_d);
    cudaFree(outputMatrix_d);
    cudaFree(filterMatrix_d);

    return 0;
}