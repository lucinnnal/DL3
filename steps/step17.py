import weakref
import numpy as np

# Variable Class를 투명한 데이터로 만드는 작업을 실행 -> ndarray 인스턴스를 담은 상자를 마치 ndarray 인스턴스를 다루는 것과 같이 만들어주는 작업

"""
추가한 코드
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
    
    # print(Variable)을 통해 객체의 정보를 문자열로 반환
    def __repr__(self):
        if self.data is None:
            return "variable(None)"
        
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

- len(obj) → 내부적으로 obj.__len__() 실행
- __len__() 메서드는 반드시 정수를 반환해야 하며, 그렇지 않으면 TypeError가 발생합니다.
- __len__() 메서드는 주로 list, tuple, str, dict 같은 시퀀스나 컬렉션 객체에 정의되어 있음

원하는 동작을 사용자 정의 클래스에 맞게 지정하려면 __len__ 메서드를 직접 구현하면 됩니다.
"""

"""
print(obj)? -> __str__?, __repr__?
좋은 질문입니다. 간단히 정리하면 다음과 같습니다:

⸻

print(객체)를 호출할 때 내부적으로 어떤 메서드가 호출되는가?
	•	print(obj)는 내부적으로 **str(obj)**를 호출합니다.
	•	str(obj)는 obj.__str__() 메서드를 먼저 찾고, 없으면 obj.__repr__()을 대신 호출합니다.

⸻

즉, 정리하면:

상황	호출되는 메서드
print(obj)	obj.__str__() → 없으면 obj.__repr__()
repr(obj)	obj.__repr__()

⸻

예시로 보면:

class MyClass:
    def __str__(self):
        return "This is str"

    def __repr__(self):
        return "This is repr"

obj = MyClass()

print(obj)       # This is str
print(repr(obj)) # This is repr


⸻

만약 __str__을 정의하지 않으면?

class MyClass:
    def __repr__(self):
        return "This is repr"

obj = MyClass()

print(obj)       # This is repr ← __str__ 없으면 __repr__ 사용


⸻

결론
	•	print(obj) → obj.__str__() 호출
	•	__str__()이 없으면 → __repr__()이 대신 호출됨

따라서 직접적으로는 __str__(), 간접적으로는 __repr__()이 호출될 수도 있습니다.

@property # @property를 사용하면 객체의 메서드를 인스턴스 변수처럼 사용 가능, x.shape()가 아니라 x.shape
def shape(self):
    return self.data.shape # ndarray 객체의 shape라는 인스턴스 변수 반환
"""


"""
오버로딩과 오버라이딩
물론이죠! super에 대한 설명도 포함하여 지금까지 배운 내용 전체를 완성형으로 정리해드릴게요.

⸻

🧠 지금까지 정리한 내용

⸻

✅ 1. 특수 메서드 (Magic / Dunder Methods)

메서드	역할
__len__(self)	len(obj) 호출 시 실행됨
__repr__(self)	repr(obj) 또는 디버깅 시 객체 정보 문자열 반환
__str__(self)	print(obj) 또는 str(obj) 호출 시 실행됨→ 없으면 __repr__()이 호출됨

사용자 정의 클래스도 이 메서드들을 정의하면, 마치 기본 자료형처럼 작동하게 만들 수 있음.

⸻

✅ 2. 연산자 오버로딩 (Operator Overloading)

+, *, == 같은 연산자를 클래스에 맞게 동작하도록 만드는 것.

연산자	메서드
+	__add__
-	__sub__
*	__mul__
==	__eq__
()	__call__
[]	__getitem__

🔸 예

class Vector:
    def __init__(self, x): self.x = x
    def __mul__(self, other):
        return Vector(self.x * other.x)

a = Vector(2)
b = Vector(3)
c = a * b  # 내부적으로 a.__mul__(b)

또는 외부 함수로 mul(x, y)을 정의한 후,

Variable.__mul__ = mul

처럼 동적으로 오버로딩 메서드를 할당할 수도 있음.

⸻

✅ 3. 함수 오버로딩 (Function Overloading)

같은 이름의 함수가 다른 인자 수/타입에 따라 다르게 작동하는 것

	•	❌ 파이썬은 직접 지원하지 않음
	•	✅ 대체 방법:
	•	*args, **kwargs로 구현
	•	@singledispatch / @singledispatchmethod 사용 (Python 3.8+)

🔸 예

def greet(*args):
    if len(args) == 1:
        print("Hello", args[0])
    elif len(args) == 2:
        print(f"Hello {args[0]}, you're {args[1]} years old")


⸻

✅ 4. 메서드 오버라이딩 (Method Overriding)

부모 클래스에서 정의한 메서드를 자식 클래스에서 재정의하는 것

🔸 예

class Animal:
    def speak(self):
        print("Animal speaks")

class Dog(Animal):
    def speak(self):  # 부모 메서드 오버라이딩
        print("Dog barks")


⸻

✅ 5. super()의 역할

super()는 부모 클래스의 메서드를 자식 클래스에서 호출할 때 사용하는 내장 함수

🔸 예: 오버라이딩된 부모 메서드도 호출하고 싶을 때

class Dog(Animal):
    def speak(self):
        super().speak()       # 부모의 speak() 호출
        print("Dog barks")    # 추가 동작

	•	super()는 상속 계층에서 바로 위 부모 클래스의 메서드를 호출함
	•	다중 상속에서도 MRO(Method Resolution Order)를 따르므로 안정적임

⸻

🔁 오버로딩 vs 오버라이딩 요약 비교

구분	오버로딩 (Overloading)	오버라이딩 (Overriding)
의미	같은 이름, 인자 다른 함수 여러 개 정의	부모 메서드를 자식이 재정의
지원 여부	❌ 직접 지원 X (*args, @singledispatch로 대체)	✅ 완전 지원
대표 예	greet(name) vs greet(name, age)	Animal.speak() → Dog.speak()
관련 키워드	*args, @singledispatch	super()


⸻

필요하다면 super() 내부 동작 원리(MRO), @override 데코레이터 (Python 3.12+), 혹은 __call__, __getitem__ 등 특수 메서드 정리도 해드릴게요!
"""


def as_array(data):
    if not isinstance(data, np.ndarray):
        return np.array(data) # 0차원 ndarray에 대한 계산 결과값이 np.float32, np.float64가 나올 것을 대비하여 ndarray 타입으로 강제적으로 변환해주는 처리
    return data

class Variable:
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

class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0

def mul(x0, x1):
    return Mul()(x0, x1)

def add(x0, x1):
    return Add()(x0, x1)

def square(x):
    return Square()(x)

# Variable 클래스 안에서 __mul__을 정의했지만, 아래와 같이 해도 무관
Variable.__mul__ = mul
Variable.__add__ = add

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
    a = Variable(np.array(3.0))
    b = Variable(np.array(2.0))
    c = Variable(np.array(1.0))

    y = a * b + c # instance a의 __mul__ method가 호출됨, (a * b)의 __add__ 메서드가 호출됨
    y.backward()

    print(y)
    print(a.grad)
    print(b.grad)