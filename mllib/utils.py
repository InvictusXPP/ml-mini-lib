#save/load, gradient check

import pickle
import numpy as np
from .backend import asnumpy, get_xp
from .training import backprop_simple
from .tensor_ops import mse_loss


def gradient_check(
    ffn,
    X,
    y,
    epsilon: float = 1e-4,
    tol: float = 1e-4,
):
    W1 = asnumpy(ffn.hidden.W).copy()
    b1 = asnumpy(ffn.hidden.b).copy()
    W2 = asnumpy(ffn.output.W).copy()
    b2 = asnumpy(ffn.output.b).copy()

    def loss_fn():
        _, _, _, A2 = ffn.forward(X)
        return float(mse_loss(A2, y, np))

    loss, dW1, db1, dW2, db2 = backprop_simple(ffn, X, y, np)

    grads = {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
    }

    def rel_error(a, b):
        return np.max(
            np.abs(a - b)
            / np.maximum(1e-8, np.abs(a) + np.abs(b))
        )

    errors = {}

    for name, param in zip(
        ["W1", "b1", "W2", "b2"], [W1, b1, W2, b2]
    ):
        grad_num = np.zeros_like(param)
        it = np.nditer(param, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = param[idx]

            param[idx] = orig + epsilon
            ffn.hidden.W, ffn.hidden.b = W1, b1
            ffn.output.W, ffn.output.b = W2, b2
            l1 = loss_fn()

            param[idx] = orig - epsilon
            l2 = loss_fn()

            param[idx] = orig
            grad_num[idx] = (l1 - l2) / (2 * epsilon)
            it.iternext()

        errors[name] = rel_error(grads[name], grad_num)

    ok = all(e < tol for e in errors.values())
    return ok, errors


def save_params(ffn, path: str):
    data = {
        "W1": asnumpy(ffn.hidden.W),
        "b1": asnumpy(ffn.hidden.b),
        "W2": asnumpy(ffn.output.W),
        "b2": asnumpy(ffn.output.b),
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_params(ffn, path: str, backend: str = "cpu"):
    with open(path, "rb") as f:
        data = pickle.load(f)

    xp = get_xp(backend)
    ffn.hidden.W = xp.asarray(data["W1"])
    ffn.hidden.b = xp.asarray(data["b1"])
    ffn.output.W = xp.asarray(data["W2"])
    ffn.output.b = xp.asarray(data["b2"])
