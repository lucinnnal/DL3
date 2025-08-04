import numpy as np
from dezero import Variable

def sphere(x, y):
    return x ** 2 + y ** 2

if __name__ == "__main__":
    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))
    z = sphere(x, y)
    z.backward()
    print(x.grad, y.grad)