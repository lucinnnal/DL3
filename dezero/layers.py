import weakref
import numpy as np
import dezero.functions as F
from dezero import Parameter, Variable

# Layer class의 self._params에 Parameter 인스턴스에 속한 매개변수를 보관(정확히는 그 이름을 보관). 매개변수는 따로 Layer의 인스턴스 변수로 저장. -> layer = Layer()에서 layer.__dict__에 매개변수 이름과 value가 저장됨
# __setattr__(self, name, value) overriding : name 이라는 인스턴스 변수에 value를 지정(이거는 원래 파이썬 클래스 __setattr__를 호출 super를 통해 수행, 그 전에 value가 Parameter 인스턴스인지 확인 후 self._params에 파라미터 이름을 넣는 작업을 추가함).
# ex) layer.p1 = Parameter(np.array(1.0))이면 파라미터 이름 layer._params에 추가 및 layer.__dict__['p1']에 Parameter(np.array(1.0)) 인스턴스 저장
# 아래 둘은(21, 22번째 줄) __setattr__에서 isinstance가 False가 되어 layer._params의 set에는 이름이 저정되지 않음(매개 변수로 인식하지는 않을 것임), 그러나 if문 밖에서 super().__setattr__를 호출하기에 인스턴스 변수로 저장되긴함.
# Parameter 클래스라면 self._params에 그 이름을 따로 저장
# 인스턴스 변수와 그 이름은 클래스의 __dict__변수에 다 저장되어 있음.

class Layer: # Layer Class => 원하는 연산 지원 및 + Parameter 클래스를 모아두고 관리하는 계층(clear grad, 개입되는 파라미터 조회 및 파라미터 업데이트까지 지원)
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)
    
    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()
    
    # Layer에서 관리하는 실제 parameter 값을 차례로 반환하는 generator 함수, yield를 사용하여서 반환 후 함추를 종료하는 것이 아닌 함수 일시중지후 다음 yield를 기다림
    def params(self):
        for name in self._params:
            obj = self.__dict__[name]

            # Layer안에 또다른 Layer가 있을 떄 그 Layer안에 있는 params generator 호출(여러 layer가 관리하는 파라미터를 하나의 layer로 관리하기 위해 일종의 container 같은 것을 만들어놓음)
            if isinstance(obj, Layer):
                yield from obj.params() # yield from : Generator in Generator 호출 시에(다른 곳에서 generator를 호출할 시에 사용)
            else:
                yield obj

    def cleargrads(self):
        for param in self.params(): # generator 함수를 반복 호출하여 yield 값을 param이 참조할 수 있게함.
            param.cleargrad()

# Layer 클래스를 상속 받아서 원하는 선형 변환을 할 수 있도록(Layer를 상속받음으로써 파라미터 관리도 할 수 있음)
class Linear(Layer):
    def __init__(self, out_size, in_size=None, no_bias=False, dtype=np.float32):
        super().__init__() # Layer의 init 메서드 호출 => 파라미터 set 초기화
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W') # 부모인 layer의 __setattr__가 호출됨 => Parameter 인스턴스인지 확인 후, 파라미터 set에 이름 저장 및 속성 값 할당
        if self.in_size is not None: # in_size가 이미 지정되어있다면 W 초기화 실행
            self._init_W()
        
        if no_bias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=self.dtype), name='b')
    
    def _init_W(self): # parameter 초기화 함수
        I = self.in_size
        O = self.out_size
        self.W.data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
    
    def forward(self, x): # 초기화 시에 in_size를 지정 하지 않아도 forward시에 데이터의 dim=1 차원에 따라 유동적으로 shape을 결정
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()
        
        y = F.linear(x, self.W, self.b)
        return y