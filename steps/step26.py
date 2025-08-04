# Linear Regression
import numpy as np
from dezero import Variable
import dezero.functions as F

def predict(x, b, W):
    y = F.matmul(x, W) + b # b를 더할 때 자동으로 broadcasting이 일어남.
    return y

def mean_squared_error_simple(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff) # F.sum의 경우 축 지정이 none이기에 모든 원소를 더함

if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = 5 + 2 * x + np.random.rand(100, 1) # add noise

    x, y = Variable(x), Variable(y) # Function class 내부적으로 Variable 객체가 아니라면, 변환해주는 as_variable 함수를 구현했음.

    W = Variable(np.zeros((1,1)))
    b = Variable(np.zeros(1))

    lr = 0.1
    iters = 100

    for i in range(iters):
        y_pred = predict(x, b, W)
        loss = F.mean_squared_error(y_pred, y)

        # Parameter들에 대한 gradient 누적 방지를 위해 cleargrad 메서드 사용하여 gradient 초기화
        W.cleargrad()
        b.cleargrad()
        loss.backward() # Backward

        # data(nd.array 인스턴스)만 뽑아서 갱신 -> 매개 변수 갱신을 하는데는 굳이 계산그래프(define-by-run)를 만들지 않아도 되기 때문에 !
        W.data -= lr * W.grad.data
        b.data -= lr * b.grad.data

        print(W, b, loss)