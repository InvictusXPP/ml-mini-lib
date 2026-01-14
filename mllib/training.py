#train(), backprop

import time
import numpy as np

from .backend import get_xp, to_backend, asnumpy
from .tensor_ops import (
    mse_loss,
    mse_grad,
    sigmoid_grad_from_output,
    tanh_grad_from_output,
)
from .optimizers import SGD, SGDMomentum, Adam


def backprop_simple(ffn, X, y, xp):
    Z1 = X @ ffn.hidden.W + ffn.hidden.b
    A1 = ffn.hidden.activation(Z1, xp)
    Z2 = A1 @ ffn.output.W + ffn.output.b
    A2 = ffn.output.activation(Z2, xp)

    loss = mse_loss(A2, y, xp)

    dA2 = mse_grad(A2, y, xp)
    dZ2 = dA2 * sigmoid_grad_from_output(A2, xp)
    dW2 = A1.T @ dZ2
    db2 = xp.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ ffn.output.W.T
    dZ1 = dA1 * tanh_grad_from_output(A1, xp)
    dW1 = X.T @ dZ1
    db1 = xp.sum(dZ1, axis=0, keepdims=True)

    return loss, dW1, db1, dW2, db2


def train(
    ffn,
    X_np,
    y_np,
    epochs: int = 1000,
    lr: float = 0.1,
    backend: str = "cpu",
    optimizer: str = "sgd",
    batch_size=None,
    verbose: bool = True,
):
    xp = get_xp(backend)
    X = to_backend(X_np, xp)
    y = to_backend(y_np, xp)
    N = X.shape[0]

    if not ffn._params_initialized:
        ffn.init_params()

    if optimizer == "sgd":
        opt = SGD(lr)
    elif optimizer == "momentum":
        opt = SGDMomentum(lr, backend=backend)
    elif optimizer == "adam":
        opt = Adam(lr, backend=backend)
    else:
        raise ValueError("Unknown optimizer")

    losses = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(N)
        batches = (
            [idx]
            if batch_size is None
            else [idx[i:i + batch_size] for i in range(0, N, batch_size)]
        )

        epoch_loss = 0.0

        for b in batches:
            Xb = to_backend(X_np[b], xp)
            yb = to_backend(y_np[b], xp)

            loss, dW1, db1, dW2, db2 = backprop_simple(
                ffn, Xb, yb, xp
            )

            params_grads = [
                (ffn.output.W, dW2),
                (ffn.output.b, db2),
                (ffn.hidden.W, dW1),
                (ffn.hidden.b, db1),
            ]

            opt.step(params_grads)
            epoch_loss += float(asnumpy(loss)) * (len(b) / N)

        losses.append(epoch_loss)

        if verbose and epoch % max(1, epochs // 10) == 0:
            print(
                f"[{backend.upper()}] "
                f"epoch {epoch}/{epochs}, "
                f"loss={epoch_loss:.6f}"
            )

    return {
        "losses": losses,
        "time": time.time() - t0,
        "model": ffn,
    }
