import os

# os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
import time


write = False

N = 2048
A = np.random.randn(N, N).astype(dtype=np.float32)
B = np.random.randn(N, N).astype(dtype=np.float32)

if write:
    C = A @ B
    C = C.astype(dtype=np.float32)

    A = A.flatten()
    B = B.flatten()
    C = C.flatten()

    with open("gemm-out", "wb") as f:
        f.write(A)
        f.write(B)
        f.write(C)

    print(A[0], A[1], A[-1])
    print(B[0], B[1], B[-1])
    print(C[0], C[1], C[-1])

    exit()

A = torch.from_numpy(A).to("cuda").detach()
B = torch.from_numpy(B).to("cuda").detach()

print(A.dtype)

start = time.time()
with torch.no_grad():
    C = A @ B
end = time.time()

print((2 * N ** 3) / 1.0e9 / (end - start), "GFLOPS")

