#add, dot, activation, loss

def sigmoid(z, xp):
    return 1.0 / (1.0 + xp.exp(-z))

def sigmoid_grad_from_output(out, xp):
    return out * (1.0 - out)

def tanh_act(z, xp):
    return xp.tanh(z)

def tanh_grad_from_output(out, xp):
    return 1.0 - out * out

def mse_loss(pred, target, xp):
    diff = pred - target
    return 0.5 * xp.mean(diff * diff)

def mse_grad(pred, target, xp):
    return (pred - target) / pred.shape[0]
