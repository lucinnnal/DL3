import weakref
import numpy as np
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



def as_variable(obj): # 여기서 obj는 nd.array 인스턴스
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

def as_array(data):
    if not isinstance(data, np.ndarray):
        return np.array(data) # 0차원 ndarray에 대한 계산 결과값이 np.float32, np.float64가 나올 것을 대비하여 ndarray 타입으로 강제적으로 변환해주는 처리
    return data


class Variable:
    __array_priority__ = 200 # nd.array 연산자보다 우선순위가 되도록 함 np.array([1.0]) + Variable(np.array(2.0))인 상황에서 np.array의 __add__ 연산자보다 Variable의 __radd__연산자가 우선 호출됨 
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray): 
                raise TypeError(f"{type(data)}는 지원하지 않는 데이터 타입.")

        self.data = data
        self.creator = None
        self.grad = None
        self.name = name # 변수 식별을 위한 이름 추가
        self.generation = 0 # 만들어지는 단계?를 기록하는 인스턴스 변수

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1 # creator의 출력인 variable은 creator의 generation보다 하나 큼(자식)
    
    def cleargrad(self):
        self.grad = None
    
    # 속성 추가!
    @property # @property를 사용하면 객체의 메서드를 인스턴스 변수처럼 사용 가능, x.shape()가 아니라 x.shape
    def shape(self):
        return self.data.shape # ndarray 객체의 shape라는 인스턴스 변수 반환
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def __len__(self):
        return len(self.data) # ndarray 인스턴스의 __len__함수 호출
    
    def __mul__(self, other):
        return mul(self, other)
    
    # print(Variable)을 통해 객체의 정보를 문자열로 반환
    def __repr__(self):
        if self.data is None:
            return "variable(None)"
        
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'
    
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
        inputs = [as_variable(x) for x in inputs]
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

def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        return  2 * self.inputs[0].data * gy

def square(x):
    return Square()(x)

class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

class Neg(Function):
    def forward(self, x0):
        return -x0
    
    def backward(self, gy):
        return -gy

def neg(x0):
    return Neg()(x0)

class Sub(Function):
    def forward(self, x0, x1):
        return x0 - x1

    def backward(self, gy):
        return gy, -gy
    
def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)

# Variable 클래스 안에서 __mul__을 정의했지만, 아래와 같이 해도 무관
class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return div(x1, x0)

class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c

        gx = c * x ** (c - 1) * gy
        return gx

def pow(x, c):
    return Pow(c)(x)

def setup_variable():
    # Variable Class 연산자 오버로딩
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow