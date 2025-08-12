import numpy as np
import dezero
from dezero.core import Variable
from dezero.core import Function
from dezero.core import as_variable
from dezero import utils

# Simple Rule : (1) Function Class 정의 (forward, backward 함수 overriding) (2) Function 생성 한 즉시 __call__을 호출하는 메서드 구현
# 각각의 입력과 출력 타입이 무엇인지 유의하며 함수를 작성

"""
forward에 사용되는 인자 -> ndarray 인스턴스 Variable의 data들의 리스트가 언패킹되어 forward 함수로 입력
backward 인자 -> output.grad(Variable 인스턴스)들이 리스트로 묶여 언패킹되어 backward 함수로 입력, 로컬 gradient를 계산하기 위한 입력들도 Variable instances
Backward 계산 시에 이용되는 Variable Instance들은 연산자 오버로딩이 구현되어 있기에 이들의 연산 결과도 Variable Instance가 될 수 있도록 함
이렇게 backward를 통해 input의 gradient들을 게산하고 튜플 형태로 묶어서 반환 -> 이건 Function의 __call__에서 실행
"""
class Log(Function):
    def forward(self, x):
        y = np.log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx


def log(x):
    return Log()(x)

class Cos(Function):
    def forward(self, x):
        return np.cos(x)
    
    def backward(self, gy):
        x0 = self.inputs[0]
        gx = gy * -sin(x0)
        return gx

def cos(x):
    return Cos()(x)

class Sin(Function):
    def forward(self, x):
        return np.sin(x)

    def backward(self, gy):
        x0 = self.inputs[0]
        gx = gy * cos(x0)
        return gx

def sin(x):
    return Sin()(x)

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx

def exp(x):
    return Exp()(x)

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape # 변환할 형상 초기화 함수에 저장
    
    def forward(self, x): # x는 ndarray 인스턴스
        self.x_shape = x.shape # 원래 입력의 형상 저장
        y = x.reshape(self.shape) # ndarray 인스턴스에 대한 reshape 메서드 (shape 정보로 리스트, 튜플 또는 가변인자 모두 처리 가능)
        return y
    
    def backward(self, gy):
        return reshape(gy, self.x_shape) # upstream gradient를 input의 shape에 맞게 reshape만 해서 전달, backward시에 np의 reshape을 사용하지 않는 이유는 upstream gradient가 Variable 인스턴스 이기 때문

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy) # np.transpose를 사용하지 않는 이유는 gy가 Variable 인스턴스 이기 때문

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)

def transpose(x, axes=None):
    return Transpose(axes)(x)

# x를 입력받아 shape으로 복제하는 함수 
# broadcast는 원소 복사가 일어나는 함수이니까 영향을 복사가 일어난 만큼 주는 함수 -> backward는 원래 형상대로 gradient를 전달하지만 복사가 일어난만큼 gradient를 반영(더함)
# sum_to(x, shape)이라는 dezero 함수를 구현할 것임 -> shape의 형상에 맞게 텐서를 sum하는 함수 
# sum_to의 경우 원래의 입력 형상으로 gradient를 입력 형상에 맞게 복제하여 전달해야함(이때 broadcast_to가 이용됨)
"""
ex)
x = np.array([1,2,3])
y = np.boradcast_to(x, (2,3))
print(y)

y -> [[1,2,3], [1,2,3]]
"""
class Broadcast_To(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Broadcast_To(shape)(x)

# 탠서를 지정한 shape에 맞게 sum을 하는 함수, backward의 경우 입력의 형상으로 broadcasting하여 gradient 전달
class Sum_To(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Sum_To(shape)(x)

# Forward : 탠서에 대해 모든 원소의 합을 구함, Backward : 입력 탠서와 동일한 형상의 gradient를 전파
class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis = self.axis, keepdims=self.keepdims) 
        # axis -> 어느 축으로의 sum?(방향), None이라면 방향 지정 없이 모든 원소의 합(스칼라)
        # keepdims -> sum 후에 축의 개수를 유지할 것인지 
        # ((2,3) sum(axis=None)이면 축의 개수 0인 scalar인데 keepdims = True라면 (1,1)로 sum 연산 후에도 축의 개수 유지)

        return y
    
     # broadcast_to를 이용하여 입력의 형상과 맞게 gradient를 전달, broadcast_to는 추후에 구현 예정
    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx
    
def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

class MatMul(Function):
    def forward(self, x, W):
        y = x @ W # forward시에는 x, W가 ndarray 인스턴스이므로 넘파이 dot 연산 이용 = np.dot(x, W)는 x,W가 다른 타입(리스트..)이어도 지원, x.dot(W)과 x @ W는 ndarray일 때 지원(x.dot의 경우 x에 인스턴스 메서드로 dot이 있어야함)

        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gw = matmul(x.T, gy)

        return gx, gw

def matmul(x, W):
    return MatMul()(x, W) # 객체 생성과 동시에 __call__함수 호출

class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff**2).sum() / len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0

        return gx0, gx1

def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)

# Linear Function & Class
def linear_simple(x, W, b=None):
    t = matmul(x, W)
    if b is None:
        return t
    
    y = t + b
    t.data = None # 중간 계산 결과에 대한 data는 메모리에서 삭제 (add 연산의 역전파에서 t의 data 자체는 역전파 계산에 이용이 되지 않음 -> upstream gradient를 그냥 흘려 보내기 때문)

    return y

class Linear(Function):
    def forward(self, x, W, b=None):
        y = x.dot(W) # x, W, b are all ndarray instance
        if b is None:
            return y
        
        y = y + b
        return y

    def backward(self, gy): # in backward gy is a variable instance
        x, W, b = self.inputs
        gx = matmul(gy, W.T)
        gw = matmul(x.T, gy)
        gb = None if b.data is None else sum_to(gy, b.shape) # sum_to를 통해 forward시에 broadcasting된 b에 대한 gradient를 합해서 역전파(b.data.shape = b.grad.shape이어야 하므로)

        return gx, gw, gb
    
def linear(x, W, b=None):
    return Linear()(x, W, b)

# <======================== Sigmoid Function & Class ============================>
def sigmoid_simple(x): # 이도 역시 최종 sigmoid 값을 계산하기 위해 variable에 대해 중간 연산들에 대한 계산 그래프가 만들어짐 -> Variable들을 다 메모리에 가지고 있음
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y

# 오직 입력 x와 출력만을 Variable로써 메모리에 가지고 있도록 Sigmoid class와 Function 구현 (forward 함수안에서 ndarray들에 대해 내부적으로 일어나는 자잘한 연산에 대한 중간 결과값들(반환되지 않는 함수 내의 지역변수들)들은 forward가 종료되는 순간 메모리에서 삭제됨
class Sigmoid(Function):
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx

def sigmoid(x):
    return Sigmoid()(x)

class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        xp = dezero.cuda.get_array_module(gy)
        gx = xp.zeros(self.in_shape, dtype=gy.dtype)

        if xp is np:
            np.add.at(gx, self.slices, gy)
        else:
            xp.scatter_add(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)


def get_item(x, slices):
    f = GetItem(slices)
    return f(x)

class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        y = x - x.max(axis=self.axis, keepdims=True)
        y = np.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx

def softmax(x, axis=1):
    return Softmax(axis)(x)

class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        y = np.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx

def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)

def softmax_cross_entropy_simple(x, t):
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]
    p = softmax(x)
    p = clip(p, 1e-15, 1.0)  # To avoid log(0)
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data] # Fancy Indexing => 2차원 배열에서 리스트(or array)를 넘겨주는 방식으로 인덱싱 하는 방법 -> 각 데이터 당 실제 정답에 대해 모델이 출력한 확률을 인덱싱
    breakpoint()
    y = -1 * sum(tlog_p) / N
    return y

class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax(x)
        # convert to one-hot
        t_onehot = np.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y

def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)

if __name__ == "__main__":
    x = Variable(np.array([1,2,3,4,5,6]))
    y = sum(x)
    y.backward()
    print(x.grad)