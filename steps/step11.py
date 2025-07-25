"""
역전파에서의 가변 인수를 처리할 수 있도록 개선.
입력이 gradient 여러개일 때와 upstream gradient도 여러 개일 때?! -> backward에서도 데이터만 처리할 수 있도록 개선
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
        while creators:
            creator = creators.pop()
            # Upstream Gradients
            gys = [output.grad for output in creator.outputs]
            # Input Gradients
            gxs = creator.backward(*gys)

            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for input, gx in zip(creator.inputs, gxs):
                input.grad = gx

                if input.creator is not None:
                    creators.append(input.creator)

            """
            하나의 Input과 하나의 Output 대응일 때
            input, output = creator.input, creator.output
            input.grad = creator.backward(output.grad)
            """
        
class Function:
    def __call__(self, *inputs): # inputs는 여러 개의 Variable Class로 이루어진 리스트였지만 이제 리스트로 받지 않게 할것임. *를 통해 가변인수들을 유연하게 처리. * 함수의 정의부에서는 가변 길이의 인자들을 튜플로 묶어서 함수에 전달함!
        xs = [x.data for x in inputs]
        outs = self.forward(*xs) # 함수 호출부에는 *가 언패킹 연산자로 작동
        if not isinstance(outs, tuple):
            outs = (outs,) # Function class의 call 메서드에서 forward의 출력이 튜플 형태가 아닐 경우 튜플로 변환하여 다음 줄을 원활히 진행할 수 있도록 함.
        outputs = [Variable(as_array(out)) for out in outs]

        # output들에 대한 생성자 정의
        for out in outputs: out.set_creator(self)

        self.inputs = inputs
        self.outputs = outputs

        return outputs if len(outputs) > 1 else outputs[0] # Function의 call 메서드 출력 시에도 리스트의 원소가 하나일 경우에는 그 값만 뱉도록 -> 리스트로 안나오게 처리, 여러 개인 경우에는 리스트를 그대로 출력

    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self):
        raise NotImplementedError()
    
class Add(Function): # 두개의 인자를 더하는 함수 
    def forward(self, x0, x1): # forward 메서드의 경우 입력과 출력을 리스트, 튜플과 같은 자료구조의 형태말고 값을 입력 출력하도록 !
        y = x0 + x1

        return y
    
    def backward(self, gy):
        return gy, gy # 각 입력에 대한 편미분 값들이 모두 1 * upstream gradient

class Square(Function):
    def forward(self, x):
        y = x ** 2

        return y
    
    def backward(self, gy):
        input = self.inputs[0].data
        gx = 2 * input * gy
        return gx

def add(x0, x1):
    return Add()(x0, x1)

def square(x):
    return Square()(x)

if __name__ ==  "__main__":
    x = Variable(np.array(2.0))
    y = Variable(np.array(3.0))

    z = add(square(x), square(y))
    z.backward()
    print(z.data)
    print(x.grad)
    print(y.grad)