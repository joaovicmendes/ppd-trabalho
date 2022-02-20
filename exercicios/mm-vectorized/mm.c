#include <stdio.h>

float matrix_a[1024][1024];
float matrix_b[1024][1024];
float result_matrix[1024][1024];

int main(int argc, char **argv) 
{
    int i, j, k;

    for (i = 0; i < 1024; i++)
        for (j = 0; j < 1024; j++) {
            matrix_a[i][j] = 0.1f;
            matrix_b[i][j] = 0.2f;
            result_matrix[i][j] = 0.0f;
        }

    #pragma omp for private(j, k)
    for (i = 0; i < 1024; i++)
        for (j = 0; j < 1024; j++)
            for (k = 0; k < 1024; k++)
                result_matrix[i][j] += matrix_a[i][k] * matrix_b[j][k];

    return 0;
}