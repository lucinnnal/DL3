import numpy as np
from dezero import Variable, Parameter
import dezero.layers as L
import dezero.functions as F

if __name__ == "__main__":
    """
    # Dataset
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    l1 = L.Linear(10) # output dimension is 10
    l2 = L.Linear(1)

    def predict(x):
        y = l1(x)
        y = F.sigmoid(y)
        y = l2(y)
        return y
    
    lr = 0.2
    iters = 10000

    for i in range(iters):
        y_pred = predict(x)
        loss = F.mean_squared_error(y, y_pred)

        # 각각의 layer가 관리하는 파라미터에 대한 gradient 초기화
        l1.cleargrads()
        l2.cleargrads()
        loss.backward()

        for l in [l1, l2]:
            for p in l.params():
                p.data -= lr * p.grad.data # grad도 Variable 인스턴스임

        if i % 100 == 0:
            print(np.round(loss.data, 4))
    """
    
    """
    각각의 linear들이 각 linear에 대한 파라미터만 관리하기에 linear 단위에서의 update와 cleargrads를 각각 진행해줘야함.
    신경망의 깊이가 깊어지면 위의 과정도 버거움. 여러 Layer를 하나로 묶어서 여러 Layer의 파라미터를 한번에 관리하는 방법이 없을까?
    """

    x = Variable(np.array(3.0))
    p = Parameter(np.array(2.0))
    y = x * p # Variable instance로 취급됨 -> * 연산이 일어날 때는 p도 Variable로 취급(Parameter는 Variable을 상속받았기 때문)