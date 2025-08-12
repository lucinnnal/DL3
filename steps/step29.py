import numpy as np
from dezero import Model, Variable
import dezero.layers as L
import dezero.functions as F

# Model Defi
"""
1. Layer가 여러 Layer들을 관리하는 container 역할을 할 수 있도록 Layer Class 코드 수정하여 모델 파라미터 관리를 Layer 단위로 하지 않아도 되게함
-> 이로써 모든 layer들의 매개변수를 container layer하나로 관리할 수 있게됨
2. 이 Layer Class를 상속 받은 Model Class에 계산 그래프 시각화 메서드를 추가함
3. 이 Model Class를 상속받아, 모델 정의에 이용
"""
class TwoLayerNet(Model):
    def __init__(self, hidden_size, output_size):
        super().__init__() # Layer의 초기화 메서드 호출 -> 파라미터 name을 저장할 set 초기화
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(output_size)
    
    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y

if __name__ == "__main__":
    # Random Dataset
    np.random.seed(0)
    x = np.random.randn(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.randn(100, 1)

    import matplotlib.pyplot as plt
    plt.scatter(x, y)
    plt.show()


    # Hyperparameter
    lr = 0.2
    max_iter = 10000
    hidden_size = 10
    
    # Model
    model = TwoLayerNet(hidden_size, 1)

    # Training 
    for iter in range(max_iter):
        y_pred = model(x)
        loss = F.mean_squared_error(y_pred, y)

        model.cleargrads()
        loss.backward()

        for p in model.params():
            p.data -= lr * p.grad.data
        if iter % 100 == 0:
            print(f"{iter+1} loss : {loss.data}\n")