#ploty

import numpy as np
import matplotlib.pyplot as plt
from .backend import get_xp, to_backend, asnumpy


def plot_losses(losses_cpu, losses_gpu=None, title="Loss vs Epoch"):
    plt.figure(figsize=(8, 5))
    plt.plot(losses_cpu, label="CPU")
    if losses_gpu is not None:
        plt.plot(losses_gpu, label="GPU")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_decision_boundary(ffn, title="Decision Boundary"):
    xx = np.linspace(-0.5, 1.5, 200)
    yy = np.linspace(-0.5, 1.5, 200)
    XX, YY = np.meshgrid(xx, yy)

    pts = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)
    xp = get_xp(ffn.backend)
    pts_b = to_backend(pts, xp)

    preds = asnumpy(ffn.predict(pts_b)).reshape(XX.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(XX, YY, preds, levels=50, alpha=0.8)
    plt.scatter(
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        c=[0, 1, 1, 0],
        edgecolors="k",
    )
    plt.title(title)
    plt.show()
