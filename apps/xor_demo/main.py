import numpy as np
import cupy as cp
from mllib.model import SimpleFFN
from mllib.training import train
from mllib.backend import cupy_available

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y = np.array([[0],[1],[1],[0]], dtype=np.float32)

# ===== CPU =====
net_cpu = SimpleFFN(2, 8, 1, backend="cpu")
res_cpu = train(net_cpu, X, y, epochs=3000, lr=0.1, backend="cpu")

print("CPU preds:")
print((net_cpu.predict(X) > 0.5).astype(int))

# ===== GPU =====
if cupy_available:
    net_gpu = SimpleFFN(2, 8, 1, backend="gpu")
    res_gpu = train(net_gpu, X, y, epochs=3000, lr=0.1, backend="gpu")

    X_gpu = cp.asarray(X)

    print("GPU preds:")
    print((net_gpu.predict(X_gpu) > 0.5).astype(int))
else:
    print("GPU not available")
