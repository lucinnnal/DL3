from dezero import Layer
from dezero import utils
import dezero.functions as F
import dezero.layers as L
from typing import Union

# 기존 Layer class에 계산 그래프 시각화 코드만 추가
class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)

# 유동적으로 층의 개수 및 각 층의 hidden vector size를 정해줄 수 있도록 MLP 클래스 구성, activation function의 default는 sigmoid.
class MLP(Model):
    def __init__(self, fc_output_sizes : Union[tuple, list], activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, output_size in enumerate(fc_output_sizes):
            layer = L.Linear(output_size)
            setattr(self, 'l'+str(i), layer) # 반복문으로 layer 정의하기에 self.l1 이런식으로 지정할 수 없고 setattr를 이용해야함 => 오버라이딩 했던 __setattr__이 호출됨
            self.layers.append(layer)
        
    def forward(self, x):
        for l in self.layers[:-1]: # 마지막 layer는 activation 적용 안함
            x = self.activation(l(x))
        return self.layers[-1](x)

if __name__ == "__main__":
    model1 = MLP((10,1))
    model2 = MLP((10, 20, 30, 40, 1))