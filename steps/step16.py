import weakref
import numpy as np



"""
1.
학습 시에는 backpropagation을 위해 input 값을 저장해놓고 있어야함.
그러나 추론 시에는 역전파가 필요하지 않기 때문에 저장해놓고 있을 필요가 없음.
-> 이를 나타내는 Flag 값을 클래스 속성으로 가지는 Config 정의
-> Config의 enable_prop이 True일 때만 define-by-run 및 generation 설정을 하도록함.
self.inputs = inputs # inputs tuple 데이터에 대한 reference 수 하나 증가 (1)

2. Mode Convert (역전파 가능 모드에서 불가능 모드로 for inference)

with 문
-> 후처리를 자동으로 수행하고자 할때 쓰는 구문
f = open('sample.txt', 'w')
f.write('hello world!')
f.close()

매번 open하고 close하는 것이 귀찮고 까먹을 때도 있음
with open('sample.txt', 'w') as f:
    f.write("hello")

이런식으로 with문을 쓰면 open 후에 with 안의 구문을 끝내는 순간 자동으로 후처리(close)를 할 수 있게 해줌

이런 원리를 이용하여 역전파 비활성 모드를 with 문으로 실행하고 with 블록을 벗어나면 다시 역전파 활성모드(후처리)로 자동으로 돌아갈 수 있게 해주고 싶음!
ex) 모델 평가를 학습 도중에 하고 싶을 때!

import contextlib

@contextlib.contextmanager
def config_test():
    print('start') # 전처리
    try:
        yield
    finally:
        print('done') # 후처리

with config_test():
    print("process") # 먼저 전처리를 진행하고 난 후에 해당 함수가 호출되어 process를 출력하고 with문이 끝나는 대로 print('done') 후처리를 진행


-> with문과 contextmanager에 대해 더 자세히 알고 싶다면, https://www.notion.so/Contextmanager-with-23c9f61d80ff805a999cd371090c7046 참조
"""

"""
** 추가한 코드 **

import contextlib

class Config:
    enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name) # 다시 미분 가능 모드로 되돌려놓기 위해 저장
    setattr(Config, name ,value) # 미분 불능으로 만들기
    try:
        yield # with 문 블럭 수행이 끝날 때까지 기다림
    finally:
        setattr(Config, name, old_value) # with문 블럭이 끝나는 순간 미분 가능한 모드로 돌려놓기

def no_grad():
    return using_config('enable_backprop', False)


with using_config('enable_backprop', False):
    pass
    
아니면 

with no_grad():
    pass
    
-> 왜 이런 no_grad() 문이 필요한가? : 학습 도중 evaluation을 할 시에는 backpropagation이 필요 없음. 불필요한
Generation 설정, function의 input 및 output 설정, Variable의 creator 설정을 함으로써 메모리 아끼는 효과
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

        # Config의 enable_prop이 True일 때만 define-by-run 및 generation 설정을 하도록함.
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs]) # input의 세데 중 가장 큰 세대를 Function의 generation으로 설정
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs # inputs tuple 데이터에 대한 reference 수 하나 증가 (1)
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

import contextlib

class Config:
    enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name) # 다시 미분 가능 모드로 되돌려놓기 위해 저장
    setattr(Config, name ,value) # 미분 불능으로 만들기
    try:
        yield # with 문 블럭 수행이 끝날 때까지 기다림
    finally:
        setattr(Config, name, old_value) # with문 블럭이 끝나는 순간 미분 가능한 모드로 돌려놓기

def no_grad():
    return using_config('enable_backprop', False)

if __name__ == "__main__":
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    t = add(x0, x1)
    y = add(x0, t)
    y.backward()
    print(y.grad, t.grad)  # None None
    print(x0.grad, x1.grad)  # 2.0 1.0