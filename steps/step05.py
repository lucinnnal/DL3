# Define-By-Run : 데이터를 흘려(Run)보냄으로써, 연결(창조자와 입력 및 출력의 연결관계가 Define된다!)
# Automize the Backward Process
import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
        f = self.creator
        # Iterate until the variable does not have creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()
    
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        # Define-By-Run
        # Define Output's Creator & Function's input, output Variable
        output.set_creator(self) 
        self.input = input
        self.output = output

        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        return self.input.data * 2 * gy

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        return np.exp(self.input.data) * gy # local gradient * upstream gradient


if __name__ == "__main__":
    A = Square()
    B = Exp()
    C = Square()

    # Forward
    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    # Backward
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)