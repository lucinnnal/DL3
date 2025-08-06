import weakref
import numpy as np
import contextlib
import dezero
"""

역전파 시에도 self.grad가 Variable 인스턴스를 참조하게 함으로써 계산 그래프를 형성하도록 함 이렇게 하면 1차 미분된 gradient에 대한 입력값의 gradient (2차 미분값도 구할 수 있게됨.)

# Variable의 Backward 메서드에서...

    with using_config('enable_backprop', create_graph): # 역전파 시에 Graph를 형성 할지 말지를 여기서 결정 (create graph가 False라면 역전파시 그래프 형성 못하게 함) (input output 저장)
        # Backward 연산 일어날 시에 Function 클래스는 __call__에서 enable_backprop 플래그 값을 참조하여 input, output caching할 지를 판단 
        # 대부분의 practical한 케이스에서는 create_graph가 False : 1차 미분까지만 필요하기 때문에. 2차 이상의 미분이 필요하다면 역전파 시에도 graph를 연결하기 위해 True로 설정.

# 이 모든 과정, 역전파 시에 그래프를 정의하는 과정은 고차 미분을 구하기 위함임.

왜? 예를 들어 Newton 최적화 방법을 통해 경사하강법보다 더 빠르게 최적화를 진행하고 싶을 때!
- f(x)에 대해 임의의 점에서 시작 (a)
- 테일러 급수 전개(f(x)의 도함수들의 급수 전개로 해당 지점에서의 f(x)를 근사하는 다항식을 도출)로 해당 지점에서의 f(x)를 근사하는 다항식 도출
- 해당 지점에서 테일러 급수의 최솟값 방향으로 업데이트 하는 방식을 반복 (이 과정에서 f(x)에 대한 고차 미분값이 필요, newton 방법은 해당지점에서 이계도함수까지만 구하여 f(x)를 근사하려고 함.)

"""

# =============================================================================
# Config
# =============================================================================
class Config:
    enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)

# =============================================================================
# Variable / Function
# =============================================================================
class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape 

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
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)): # *를 통해 튜플로 묶어서 전달했을 때, 길이가 1(요소가 하나)이고 그 타입이 리스트나 튜플이면...
            shape = shape[0] # Reshape forward 함수의 shape 정보로 shape[0]을 넘겨줌 ([2,3]) or ((2,3))
        return dezero.functions.reshape(self, shape)

    # 인스턴스 메서드로 transpose 구현
    def transpose(self): # 인스턴스 메서드로 transpose 구현
        return dezero.functions.transpose(self)
    
    @property
    def T(self):
        return self.transpose() # => Variable.T로 호출하면 바로 transpose된 Variable 인스턴스가 나오도록 함. 인스턴스 변수처럼 함수를 호출
    
    # 인스턴스 메서드로 sum 구현
    def sum(self, axis=None, keepdims=False):
        return dezero.functions.sum(self, axis, keepdims)

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False, create_graph = False):
        if self.grad is None:
            # self.grad = np.ones_like(self.data) -> 역전파 시 게산 그래프 구성을 위해 Gradient도 Variable 인스턴스화
            self.grad = Variable(np.ones_like(self.data)) 

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]  # output is weakref

            with using_config('enable_backprop', create_graph): # 역전파 시에 Graph를 형성 할지 말지를 여기서 결정 (create graph가 False라면 역전파시 그래프 형성 못하게 함) (input output 저장)
                # Backward 연산 일어날 시에 Function 클래스는 __call__에서 enable_backprop 플래그 값을 참조하여 input, output caching할 지를 판단 
                # 대부분의 practical한 케이스에서는 create_graph가 False : 1차 미분까지만 필요하기 때문에. 2차 이상의 미분이 필요하다면 역전파 시에도 graph를 연결하기 위해 True로 설정.
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx # gradient 값을 계속 누적시키는 구조

                    if x.creator is not None:
                        add_func(x.creator)
            # 자동 후처리 -> with문 들어가면서 create_graph가 False 였다면 with 문 처리 후에 다시 old_value인 True로 바뀜.

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # y is weakref

# Parameter Class : 모델의 가중치들을 한번에 처리할 수 있도록 도와주는 class (업데이트 및 cleargrad)
class Parameter(Variable):
    pass

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]


        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()

# =============================================================================
# 사칙연산 / 연산자 오버로드
# =============================================================================
class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:  # for broadcaset
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:  # for broadcast
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, gy):
        gx0 = gy
        gx1 = -gy
        if self.x0_shape != self.x1_shape:  # for broadcast
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if x0.shape != x1.shape:  # for broadcast
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x, = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx

def pow(x, c):
    return Pow(c)(x)

def setup_variable():
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

if __name__ == "__main__":
    x = Variable(np.array(1.0))
    p = Parameter(np.array(2.0))

    y = x * p

    breakpoint()