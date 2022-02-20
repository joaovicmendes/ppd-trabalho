#include <stdio.h>

float matrix_a[1024][1024];
float matrix_b[1024][1024];
float result_matrix[1024][1024];

int main(int argc, char **argv) 
{
  int i,j,k;
 
  // initialize arrays
  for (i = 0; i < 1024; i++) {
    for (j = 0; j < 1024; j++) {
      matrix_a[i][j] = 0.1f;
      matrix_b[i][j] = 0.2f;
      result_matrix[i][j] = 0.0f;     // pode ser substituído por memset (0...)
    } 
  }
  #pragma omp for private(j, k)
  for (i = 0; i < 1024; i++) {     // iterate over rows of matrix A/result matrix
    for (j = 0; j < 1024; j++) {   // iterate over columns matrix B/result matrix
      for (k = 0; k < 1024; k++) { // iterate over colums of matrix A and COLUMNS of matrix BT
        result_matrix[i][j] += matrix_a[i][k] * matrix_b[j][k];
      }
    }
  }

//   for (i = 0; i < 1024; i++) {
//     for (j = 0; j < 1024; j++) {
//       printf("%f ", result_matrix[i][j]);
//     }
//     printf("\n");
//   }

  return 0;
 }