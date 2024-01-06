#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define EPS 0.001

#define N 2048
#define BLOCK 16

__global__ void _gemm(float *a, float *b, float *c) {
  int row = blockIdx.y * BLOCK + threadIdx.y;
  int col = blockIdx.x * BLOCK + threadIdx.x;

  __shared__ float a_cache[BLOCK][BLOCK];
  __shared__ float b_cache[BLOCK][BLOCK];

  float dot = 0.0;
  for (int i = 0; i < N; i += BLOCK) {
    if (i + threadIdx.x < N && row < N)
      a_cache[threadIdx.y][threadIdx.x] = a[row * N + i + threadIdx.x];
    else
      a_cache[threadIdx.y][threadIdx.x] = 0;
    if (i + threadIdx.y < N && i + threadIdx.y < N)
      b_cache[threadIdx.y][threadIdx.x] = b[(i + threadIdx.y) * N + col];
    else
      b_cache[threadIdx.y][threadIdx.x] = 0;
    __syncthreads();

    for (int j = 0; j < BLOCK; j++) {
      dot += a_cache[threadIdx.y][j] * b_cache[j][threadIdx.x];
    }
    __syncthreads();
  }
  if (col < N && row < N)
    c[row * N + col] = dot;
}

void gemm(float *a_h, float *b_h, float *c_h) {
  float *a_d, *b_d, *c_d;

  cudaMalloc(&a_d, N * N * sizeof(float));
  cudaMalloc(&b_d, N * N * sizeof(float));
  cudaMalloc(&c_d, N * N * sizeof(float));

  cudaMemcpy(a_d, a_h, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, N * N * sizeof(float), cudaMemcpyHostToDevice);

  int numBlocks = N / BLOCK;

  float time_ms;
  cudaEvent_t start, stop;

  // timing
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  _gemm<<<dim3(numBlocks, numBlocks, 1), dim3(BLOCK, BLOCK, 1)>>>(a_d, b_d,
                                                                  c_d);
  cudaDeviceSynchronize();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_ms, start, stop);

  printf("%f GFLOPS (compute)\n", (2.0 * N * N * N) / 1.0e6 / (double)time_ms);

  cudaMemcpy(c_h, c_d, N * N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
}

int main() {
  float *a = (float *)malloc(N * N * sizeof(float));
  float *b = (float *)malloc(N * N * sizeof(float));
  float *c = (float *)malloc(N * N * sizeof(float));
  float *ref = (float *)calloc(N * N, sizeof(float));

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
    if (fabs(c[i] - ref[i]) > EPS) {
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
