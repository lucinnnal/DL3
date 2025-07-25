import weakref
import numpy as np

"""
보통 입력 변수에 대한 gradient에 대해 관심이 있는 것이므로
중간 변수에 대한 gradient에 관심이 없다면 None 처리하는 코드 추가
"""

"""
파이썬은 참조 카운트 방식의 메모리 관리를 함. 
참조 카운트가 0이 되는 순간 해당 객체, 데이터에 대한 메모리 할당은 없어지는 방식으로 메모리 관리를 진행.
y가 실제 객체에 대해 약한 참조를 하고 있는 상황에서 y() 하면 실제 객체에 접근할 수 있음. 실제 객체에 대해 None을 했기 때문에 y()가 실제로 가리키는 객체의 grad 변수(미분 값) 값의 참조 카운트 수는 0이 되어 더이상 메모리에 남지 않게됨
"""

"""
a = np.array([1,2,3]) # array 객체의 참조 카운트는 1 (강한 참조)
b = weakref.ref(a) # 여전히 array 객체의 참조 카운트는 1 (약한 참조)

# 약한 참조시에 데이터를 확인하는 방법
print(b()) # 그냥 b가 아니라 b()로 하면 약한 참조를 하는 실제 객체/데이터에 접근을 할 수 있음

a = None # array 객체의 참조 카운트가 0 -> array 객체의 참조 카운트가 0이 되어서 메모리 할당 헤제됨, 만약 b = a로 했다면 array 객체에 대한 참조 카운트 수는 2에서 1로 줄어들기에 메모리 해제가 되지 않음

print(b) # dead
"""

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
    
    def backward(self, retain_grad=False):
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
            gys = [output().grad for output in f.outputs] # weakref된 객체는 ()로 접근
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
            
            # f.output들은 모두 약한 참조 되어있으므로, ()로 약한 참조한 실제 객체에 접근 -> 입력말고의 중간 변수의 gradient는 None 처리하고 싶을 때
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None # 약한 참조한(참조 카운트가 올라가지 않음) 미분 값 데이터에 대헤 None로 만들어서 미분 값 데이터에 대한 참조 수는 0으로 되어 메모리에서 삭제
                    # y가 실제 객체에 대해 약한 참조를 하고 있는 상황에서 y() 하면 실제 객체에 접근할 수 있음. 실제 객체에 대해 None을 했기 때문에 y()가 실제로 가리키는 객체의 grad 변수(미분 값) 값의 참조 카운트 수는 0이 되어 더이상 메모리에 남지 않게됨

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
        self.outputs = [weakref.ref(output) for output in outputs] # 함수의 output에 대해 약한 참조를 취하여서, 참조는 하지만 output에 대한 참조 카운트를 증가시키지 않음.

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
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    t = add(x0, x1)
    y = add(x0, t)
    y.backward()
    print(y.grad, t.grad)  # None None
    print(x0.grad, x1.grad)  # 2.0 1.0