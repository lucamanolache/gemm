#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define EPS 0.00001

#define N 2048
#define AB 8
#define BB 16 // should be 16 (cache line bytes / float bytes)

void gemm(float *restrict a, float *restrict b, float *restrict c) {
  for (int i = 0; i < N; i += AB) {
    for (int j = 0; j < N; j += BB) {

      float acc[AB][BB] = {};
      for (int k = 0; k < N; k++) {
        for (int i2 = 0; i2 < AB; i2++) {
          for (int j2 = 0; j2 < BB; j2++) {
            acc[i2][j2] += a[(i + i2) * N + k] * b[k * N + j + j2];
          }
        }
      }

      for (int i2 = 0; i2 < AB; i2++) {
        for (int j2 = 0; j2 < BB; j2++) {
          c[(i + i2) * N + j + j2] = acc[i2][j2];
        }
      }
    }
  }
}

int main() {
  float *a = aligned_alloc(64, N * N * sizeof(float));
  float *b = aligned_alloc(64, N * N * sizeof(float));
  float *c = aligned_alloc(64, N * N * sizeof(float));
  float *ref = calloc(N * N, sizeof(float));

  FILE *f = fopen("gemm-out", "rb");
  fread(a, sizeof(float), N * N, f);
  fread(b, sizeof(float), N * N, f);
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

  free(a);
  free(b);
  free(c);
  free(ref);

  return 0;
}
