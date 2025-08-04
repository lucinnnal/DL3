import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

"""
고차 미분 계산을 위해서는 backward 과정에서도 연산에 개입되는 요소(upstream gradient와 local gradient)들을 Variable 클래스로 만들어 주면 된다 원래 
Variable.grad 값은 ndarray 인스턴스를 참조 했으나, 이제 Variable.grad값도 Variable 클래스를 참조할 수 있도록 한다. P.257의 그림 31-3 참조. -> 이로써 순전파 뿐만 아니라
역전파에 대한 계산 그래프도 만들어짐.
"""


import numpy as np
from dezero import Variable
from dezero import Function

class Sin(Function):
    def forward(self, x): # method overriding
        return np.sin(x)
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx
    
def sin(x):
    return Sin()(x)

if __name__ == "__main__":
    x = Variable(np.array(1.0))
    y = sin(x)
    y.backward(retain_grad = True) # 보통 입력값의 gradient만 관심이 있는 상황이므로, 중간 출력값은 gradient 저장을 하지 않는데 여기서는 retain_grad = True로 함.
    breakpoint()
    print(x.grad)