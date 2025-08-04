# Matrix Multiplication & Backward
import numpy as np
from dezero import Variable
import dezero.functions as F # dezero의 functions 모듈을 F로 불러옴

if __name__ == "__main__":
    x = Variable(np.random.randn(2, 3))
    W = Variable(np.random.randn(3, 4))
    y = F.matmul(x, W)
    y.backward()