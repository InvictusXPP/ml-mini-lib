#simpleFFN
# Prosty feedforward network z możliwością rozszerzeń
from .backend import get_xp
from .tensor_ops import sigmoid, tanh_act
from .layers import Dense, Dropout, BatchNormSimple
from .init import init_params

class SimpleFFN:
    def __init__(self, n_in:int, n_hidden:int, n_out:int, backend='cpu', hidden_activation=tanh_act, out_activation=sigmoid, use_dropout=False, dropout_p=0.5, use_batchnorm=False):
        self.backend = backend
        self.xp = get_xp(backend)
        self.hidden = Dense(n_in, n_hidden, backend=backend, activation=hidden_activation, name='hidden')
        self.output = Dense(n_hidden, n_out, backend=backend, activation=out_activation, name='output')
        self.use_dropout = use_dropout
        self.dropout = Dropout(p=dropout_p, backend=backend) if use_dropout else None
        self.use_batchnorm = use_batchnorm
        self.bn = BatchNormSimple(n_hidden, backend=backend) if use_batchnorm else None
        self._params_initialized = False

    def init_params(self, seed=123):
        W1, b1, W2, b2 = init_params(self.hidden.W.shape[0], self.hidden.W.shape[1], self.output.W.shape[1], backend=self.backend, seed=seed)
        self.hidden.W = W1
        if b1.shape == self.hidden.b.shape:
            self.hidden.b = b1
        self.output.W = W2
        if b2.shape == self.output.b.shape:
            self.output.b = b2
        self._params_initialized = True

    def forward(self, X, train=True):
        Z1 = X @ self.hidden.W + self.hidden.b
        A1 = self.hidden.activation(Z1, self.xp) if self.hidden.activation else Z1
        if self.use_batchnorm:
            A1 = self.bn.forward(A1)
        if self.use_dropout:
            A1 = self.dropout.forward(A1, train=train)
        Z2 = A1 @ self.output.W + self.output.b
        A2 = self.output.activation(Z2, self.xp) if self.output.activation else Z2
        return Z1, A1, Z2, A2

    def predict(self, X):
        _, _, _, A2 = self.forward(X, train=False)
        return A2

    def get_weights(self):
        return (self.hidden.W, self.hidden.b, self.output.W, self.output.b)

    def set_weights(self, W1, b1, W2, b2):
        self.hidden.W, self.hidden.b, self.output.W, self.output.b = W1, b1, W2, b2