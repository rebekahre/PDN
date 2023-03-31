//CITATIONS:
//Lecture slides, class exercises

#include <stdio.h>

#define BLUR_SIZE 2

__global__ void kernel(int *inputMatrix_d, int *outputMatrix_d, int *filterMatrix_d, int n_row, int n_col)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= n_row || col >= n_col)
    {
        return;
    }

    int sum_val = 0;

    for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE +1; ++blurRow)
    {
        for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol)
        {
            int curRow = row + blurRow;
            int curCol = col + blurCol;

            int i_row = blurRow + BLUR_SIZE;
            int i_col = blurCol + BLUR_SIZE;

            if( curRow > -1 && curRow < n_row && curCol > -1 && curCol < n_col)
            {
                sum_val += inputMatrix_d[curRow*n_col + curCol]*filterMatrix_d[i_row*5 + i_col]; 
            }
        }
    }

    outputMatrix_d[row*n_col+col] = sum_val;
}