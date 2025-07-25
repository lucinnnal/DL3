import numpy as np

class Variable:
    def __init__(self, data):
        # ndarray 타입만 Variable의 data로 받을 수 있도록 -> TypeError 처리, 또한 forward 연산의 결과도 ndarray의 타입이 되도록 보장!
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} 타입은 지원되지 않습니다.')
            
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
        # 맨 처음 gradient 역전파 시에 gradient를 1로 명시적으로 설정 안하 도록(backward 메서드 안에서 처음 gradient를 1로 초기화 하도록)
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        creators = [self.creator]
        while creators:
            creator = creators.pop()
            input, output = creator.input, creator.output
            input.grad = creator.backward(output.grad)

            if input.creator is not None:
                creators.append(input.creator)

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y)) # Forward 결과로 나오는 값이 ndarray 타입이 아닌 경우도 있을 수도 있기에, as_array 함수를 이용
        # Define By Run
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
        return np.exp(self.input.data) * gy
    
# 함수 인스텐스 생성 및 호출하는 함수를 생성
def exp(x):
    return Exp()(x)

def square(x):
    return Square()(x)

"""
0차원 ndarray 타입에 연산을 진행하면 스칼라 타입으로 바뀌는 경우도 생김 x = np.array(1.0) x = x ** 2 (x는 np.float64)
np.isscalar -> np.float32 또는 np.float64 같은 스칼라 타입인지 확인
결과가 ndarray 타입이 아니어도 ndarray가 나오도록 보장하는 as_array 함수 추가
"""

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

if __name__ == "__main__":
    pass