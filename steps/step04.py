import numpy as np

# Variable with Gradient
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None

# Function with Gradient
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        self.input = input # caching input for backward calculation
        output = Variable(y)

        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy): # gy is upstream gradient
        raise NotImplementedError()

# Square & Exp Function with backward
class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        return 2 * x * gy

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        return np.exp(x) * gy

if __name__ == '__main__':
    # Forward
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    # Backward
    y.grad = 1
    b.grad = C.backward(y.grad)
    a.grad = B.backward(b.grad)
    x.grad = A.backward(a.grad)
    print(f"x_grad = {x.grad}")