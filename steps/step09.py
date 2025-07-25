"""
가변 길이의 인수와 가변 길이의 출력 -> 덧셈 같이 여러 개의 입력을 받는 함수가 있을 수도 있고 반대로 여러 개의 출력을 내는 함수도 있을 수 있음.
이를 고려하여 처리하도록 기존의 함수 클래스를 개선
"""
import numpy as np

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} 타입은 지원되지 않습니다.')
        
        self.data = data
        self.creator = None
        self.grad = None
    
    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data) # gradient가 아직 없다면 1로 초기화 (형상은 자기 자신과 같은)
        
        creators = [self.creator]
        while not creators:
            creator = creators.pop()
            input, output = creator.input, creator.output
            input.grad = creator.backward(output.grad)

            if input.creator is not None:
                creators.append(input.creator)
        
class Function:
    def __call__(self, inputs): # inputs는 여러 개의 Variable Class로 이루어진 리스트.
        xs = [x.data for x in inputs]
        outs = self.forward(xs) # output도 가변적으로 처리할 수 있도록 처리
        outputs = [Variable(as_array(out)) for out in outs]

        # output들에 대한 생성자 정의
        for out in outputs: out.set_creator(self)

        self.inputs = inputs
        self.outputs = outputs

        return outputs

    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self):
        raise NotImplementedError()
    
class Add(Function): # 두개의 인자를 더하는 함수
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1

        return (y,)

if __name__ == "__main__":
    inputs = [Variable(np.array(1.0)), Variable(np.array(2.0))]
    outputs = Add()(inputs)
    breakpoint()
