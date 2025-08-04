import numpy as np
from dezero import Variable
import dezero.functions as F

"""
모듈 : 파이썬 파일 .py
패키지 : 여러 모듈을 묶어놓은 것
라이브러리 : 여러 패키지를 묶어놓은 것 (때로는 패키지 하나를 가리켜 '라이브러리'라고 부름)


from __ import Variable 라고 쓰면 모듈 내의 클래스나 함수를 직접 임포트 할 수 있음.
import __ as A이면 ___라는 모듈을 A라는 이름으로 불러올 수 있음.
ex) import dezero.core_simple as dz이면 dezero.core_simple 모듈을 dz라는 이름으로 임포트함. 그 후 해당 모듈 안의 Variable 클래스를 사용하려면 dz.Variable이라 쓰면됌 -> import numpy as np 하고 np.array하는 것은 numpy 모듈의 array 클래스를 인스턴스화 하는 것과 같다!!

__init__.py는 패지지로부터 모듈을 호출할 때 가장 먼저 실행되는 파일임 ! from dezero.core_simple import Variable 하면 dezero/core_simple .py 모듈의 Variable을 import 하기 전에 dezero 패키지의 __init__.py가 먼저 실행된다.
from dezero import Variable이 가능한 이유!!! : 패키지가 import 될 때, __init__.py 파일이 가장 먼저 실행되고 여기서 from dezero.core_simple import Variable이 먼저 실행되기 때문
"""

if __name__ == "__main__":
    x = Variable(np.array([[1,2,3], [4,5,6]]))
    y = F.sin(x)
    print(y)
    