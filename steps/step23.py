import numpy as np
from dezero import Variable
import dezero.functions as F

"""
< Reshape 함수의 역할 >
Forward : input의 형상을 reshape함.
Backward : reshape된 형상을 원래 input의 형상으로 바꿔서 gradient 전달 -> Local Gradient는 모두 1.
reshape은 단지 인덱스를 재배열하는 연산일 뿐, 계산 자체가 없기 때문에 입력 값에 대한 도함수는 항상 1입니다.

입력이 tensor 형식이어도 입력의 data 형상과 grad 형상이 같아야한다는 것을 기억하자!

x = Variable(np.array([[1,2,3], [4,5,6]]))
y = F.reshape(x, (6,))
y.backward(retain_grad=True)
print(y.grad)
print(x.grad)

<Transpose 함수>
이것도 역시 forward 과정에서 형상만 바꿈 (축 교체)
backward시에도 transpose된 형상을 원래 input의 형상으로 바꿔 gradient 전달 -> Local Gradient는 모두 1.
"""

"""
Variable class 안에 reshape 구현 ! => Instance Method 처럼
아래와 같이 Variable.reshape도 3가지 타입의 shape 정보를 모두 처리할 수 있도록하기

x = np.random.rand(1,2,3)
# 형상 정보로 다음을 전달 가능 -> 아래의 3가지 방식
y = x.reshape(2,3) # 가변 인자 전달 가능
y = x.reshape([2,3]) # 리스트 전달
y = x.reshape((2,3)) # 튜플 

"""

if __name__ == "__main__":
    x = Variable(np.random.randn(1, 2, 3))
    y = x.reshape((2,3))
    print(y.shape)
    y = x.reshape(2,3)
    print(y.shape)

    x = Variable(np.array([[1,2,3], [4,5,6]]))
    y = F.transpose(x)
    print(y)
    y.backward() # 입력 data = 입력 gradient 같아야 되므로 원래 입력 데이터의 형상과 같은 local gradient 1이 나옴
    print(x.grad)

    x = Variable(np.array([[1,2,3], [4,5,6]]))
    y = x.transpose() # instance method로 transpose 반환
    print(y)
    y = x.T # 속성 변수처럼 transpose method 호출 (@property)
    print(y)