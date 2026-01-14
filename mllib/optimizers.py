#SGD, Momentum, Adam

from .backend import get_xp


class SGD:
    def __init__(self, lr: float = 0.1):
        self.lr = lr

    def step(self, params_grads):
        for p, g in params_grads:
            p -= self.lr * g


class SGDMomentum:
    def __init__(
        self,
        lr: float = 0.1,
        momentum: float = 0.9,
        backend: str = "cpu",
    ):
        self.lr = lr
        self.momentum = momentum
        self.backend = backend
        self.xp = get_xp(backend)
        self.vs = []

    def init_states(self, params):
        self.vs = [self.xp.zeros_like(p) for p in params]

    def step(self, params_grads):
        if not self.vs:
            self.init_states([p for p, _ in params_grads])

        for i, (p, g) in enumerate(params_grads):
            self.vs[i] = (
                self.momentum * self.vs[i]
                + (1.0 - self.momentum) * g
            )
            p -= self.lr * self.vs[i]


class Adam:
    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        backend: str = "cpu",
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.backend = backend
        self.xp = get_xp(backend)

        self.m = []
        self.v = []
        self.t = 0

    def init_states(self, params):
        self.m = [self.xp.zeros_like(p) for p in params]
        self.v = [self.xp.zeros_like(p) for p in params]

    def step(self, params_grads):
        if not self.m:
            self.init_states([p for p, _ in params_grads])

        self.t += 1

        for i, (p, g) in enumerate(params_grads):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p -= self.lr * m_hat / (self.xp.sqrt(v_hat) + self.eps)
