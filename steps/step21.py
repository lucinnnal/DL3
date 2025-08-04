import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

def rosenblock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (1 - x0) ** 2
    return y

if __name__ == "__main__":
    # Initial Point
    x0 = Variable(np.array(0.0))
    x1 = Variable(np.array(2.0))
    iters = 1000
    lr = 1e-3

    # GD
    for i in range(iters):
        print(x0, x1)
        print(x0.grad, x1.grad) # gradient는 출력을 가장 크게 하는 방향을 나타냄 -> 최적화 할 때는 그 반대 방향으로 업데이트 해야함.

        y = rosenblock(x0, x1)

        x0.cleargrad()
        x1.cleargrad()

        y.backward()

        x0.data -= x0.grad * lr
        x1.data -= x1.grad * lr
 