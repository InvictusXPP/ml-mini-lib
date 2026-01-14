import time
import numpy as np
import cupy as cp
from mllib.model import SimpleFFN
from mllib.training import train
from mllib.backend import cupy_available

def run_benchmark(X, y, epochs, hidden, label, batch_size=1024):
    print(f"\n===== BENCHMARK: {label} =====")
    print(f"Samples: {X.shape[0]}, Epochs: {epochs}, Batch: {batch_size}")

    # CPU
    net_cpu = SimpleFFN(X.shape[1], hidden, 1, backend="cpu")
    t0 = time.perf_counter()
    train(
        net_cpu, X, y,
        epochs=epochs,
        lr=0.1,
        backend="cpu",
        batch_size=batch_size,
        verbose=False
    )
    cpu_time = time.perf_counter() - t0
    print(f"CPU time: {cpu_time:.6f} s")

    if not cupy_available:
        print("GPU not available")
        return

    import cupy as cp

    # GPU
    net_gpu = SimpleFFN(X.shape[1], hidden, 1, backend="gpu")
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()

    train(
        net_gpu, X, y,
        epochs=epochs,
        lr=0.1,
        backend="gpu",
        batch_size=batch_size,
        verbose=False
    )

    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - t0

    print(f"GPU time: {gpu_time:.6f} s")
    print(f"Speedup (CPU / GPU): {cpu_time / gpu_time:.2f}x")


def benchmark_xor():
    X = np.array(
        [[0,0],[0,1],[1,0],[1,1]],
        dtype=np.float32
    )
    y = np.array(
        [[0],[1],[1],[0]],
        dtype=np.float32
    )

    run_benchmark(
        X, y,
        epochs=20,
        hidden=64,
        label="XOR (small dataset)"
    )

def benchmark_large_dataset():
    N = 50_000   # NA START
    X = np.random.rand(N, 2).astype(np.float32)
    y = ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)) \
            .astype(np.float32).reshape(-1, 1)

    run_benchmark(
        X, y,
        epochs=20,
        hidden=1024,
        batch_size=4096,
        label="Large dataset (GPU-friendly)"
    )

    
    
if __name__ == "__main__":
    benchmark_xor()
    benchmark_large_dataset()
