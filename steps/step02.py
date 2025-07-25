# Function Class gets Variable Class as input and outputs Variable class
# And the real data of Variable Class is in instance variable data
# __call__ method를 정의하면 f = Function()의 형태로 함수의 인스턴스를 변수 f에 대입하고, 후에 f()의 형태로 Function 인스턴스의 __call__ 메서드를 호출할 수 있음

from step01 import *

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)

        return output

    def forward(self, x):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2

if __name__ == '__main__':
    x = Variable(np.array(2.0))
    f = Square()
    y = f(x)

    print(type(y))
    print(y.data)