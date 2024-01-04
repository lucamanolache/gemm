#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define EPS 0.00001

#define N 2048
#define AB 1
#define BB 16 // should be 16 (cache line bytes / float bytes)
#define KB 16

void gemm(float *restrict a, float *restrict b, float *restrict c) {
  for (int k = 0; k < N; k += KB) {
    for (int i = 0; i < N; i += AB) {
      for (int j = 0; j < N; j += BB) {

        for (int i2 = 0; i2 < AB; i2++) {
          for (int k2 = 0; k2 < KB; k2++) {
            for (int j2 = 0; j2 < BB; j2++) {
              c[(i + i2) * N + j + j2] +=
                  a[(i + i2) * N + k + k2] * b[(k + k2) * N + j + j2];
            }
          }
        }
      }
    }
  }
}

int main() {
  float *a = malloc(N * N * sizeof(float));
  float *b = malloc(N * N * sizeof(float));
  float *c = calloc(N * N, sizeof(float));
  float *ref = calloc(N * N, sizeof(float));

  FILE *f = fopen("gemm-out", "rb");
  fread(a, sizeof(float), N * N, f);
  fread(b, sizeof(float), N * N, f); // assume pre transposed
  fread(ref, sizeof(float), N * N, f);
  fclose(f);

  clock_t t;
  t = clock();
  gemm(a, b, c);
  t = clock() - t;
  double time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds
  printf("%f GFLOPS\n", (2.0 * N * N * N) / 1.0e9 / (double)time_taken);

  for (int i = 0; i < N * N; ++i) {
    if (abs(c[i] - ref[i]) > EPS) {
      printf("expected %f, got %f at idx %d\n", ref[i], c[i], i);
      return -1;
    }
  }

  return 0;
}
