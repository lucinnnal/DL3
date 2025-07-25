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
        self.generation = 0 # 만들어지는 단계?를 기록하는 인스턴스 변수

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1 # creator의 출력인 variable은 creator의 generation보다 하나 큼(자식)
    
    def cleargrad(self):
        self.grad = None
    
    def backward(self):
        if self.grad is None: 
            self.grad = np.ones_like(self.data)

        # 수정
        funcs = []
        seen_set = set() # 함수가 중복 추가되어 동일한 함수에 대해 backward()가 여러 번 호출되는 것을 방지하기 위해 set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x : x.generation) # 함수를 generation에 따라 오름차순으로 정렬하여 후세대의 함수가 먼저 pop 되도록 함

                """
                f = lambda x: x ** 2 -> lambda 입력값 : 출력값 
                print(f(3))  # 9

                map 함수
                map(함수, 반복가능한객체)
                nums = [1, 2, 3, 4] 
                squared = map(lambda x: x ** 2, nums)
                print(list(squared))  # 출력: [1, 4, 9, 16]
                """
        
        add_func(self.creator)

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
                    add_func(input.creator)

class Function:
    def __call__(self, *inputs):
        # input_datas
        xs = [input.data for input in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max([x.generation for x in inputs]) # input의 세데 중 가장 큰 세대를 Function의 generation으로 설정

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

if __name__ == "__main__":
    x = Variable(np.array(2.0))
    a = square(x)
    y = add(square(a), square(a))
    y.backward()

    print(y.data)
    print(x.grad)