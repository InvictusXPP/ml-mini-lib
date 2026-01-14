from .backend import get_xp


def init_params(
    n_in: int,
    n_hidden: int,
    n_out: int,
    backend: str = "cpu",
    seed: int = 42,
    scale: float = 0.5,
):
    xp = get_xp(backend)

    if xp.__name__ == "numpy":
        import numpy as np
        rng = np.random.RandomState(seed)

        W1 = (rng.randn(n_in, n_hidden) * scale).astype(np.float32)
        b1 = np.zeros((1, n_hidden), dtype=np.float32)
        W2 = (rng.randn(n_hidden, n_out) * scale).astype(np.float32)
        b2 = np.zeros((1, n_out), dtype=np.float32)

    else:
        try:
            xp.random.seed(seed)
        except Exception:
            pass

        W1 = xp.random.randn(n_in, n_hidden).astype(xp.float32) * scale
        b1 = xp.zeros((1, n_hidden), dtype=xp.float32)
        W2 = xp.random.randn(n_hidden, n_out).astype(xp.float32) * scale
        b2 = xp.zeros((1, n_out), dtype=xp.float32)

    return W1, b1, W2, b2
