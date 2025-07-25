"""
같은 변수를 반복적으로 사용하는 계산을 할 시에 gradient 전파가 잘 되지 않는 문제
예를 들어 y = x + x가 있는데 y.backward()를 호출하면 x.grad = 2가 아니라 1이 나옴!
-> 미분 값을 덮어쓰는 문제 때문!
"""
import numpy as np

def as_array(data):
    if not isinstance(data, np.ndarray):
        return np.array(data) # 0차원 ndarray에 대한 계산 결과값이 np.float32, np.float64가 나올 것을 대비하여 ndarray 타입으로 강제적으로 변환해주는 처리
    return data

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray): 
                raise TypeError(f"{type(data)}는 지원하지 않는 데이터 타입.")

        self.data = data
        self.creator = None
        self.grad = None

    def set_creator(self, func):
        self.creator = func
    
    def cleargrad(self):
        self.grad = None
    
    def backward(self):
        if self.grad is None: 
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)

            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            # Backward 계산은 끝났고, 이제 각 input에 gradient를 부여하는 단계
            for input, gx in zip(f.inputs, gxs):
                # 같은 input에 대해 gradient가 초기화되지 않고 누적되도록 처리
                if input.grad is None:
                    input.grad = gx
                else:
                    input.grad = input.grad + gx 

                if input.creator is not None:
                    funcs.append(input.creator)

class Function:
    def __call__(self, *inputs):
        # input_datas
        xs = [input.data for input in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        
        self.inputs = inputs
        self.outputs = outputs

        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, xs):
        raise NotImplementedError()
    
    def backward(self, gys):
        raise NotImplementedError()

class Add(Function):
    def forward(self, x0, x1):
        return x0 + x1
    
    def backward(self, gy):
        return (gy, gy)

class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        return  2 * self.inputs[0].data * gy

def add(x0, x1):
    return Add()(x0, x1)

def square(x):
    return Square()(x)

if __name__ == '__main__':
    x = Variable(np.array(3.0))
    y = add(x, x)
    y.backward()
    print(x.grad)

    x.cleargrad()
    y = add(add(x, x), x)
    y.backward()
    print(x.grad)