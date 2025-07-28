is_simple_core = True # 어디서 필요한 함수 및 클래스를 import할 지 결정

if is_simple_core:
    from dezero.core_simple import Variable
    from dezero.core_simple import Function
    from dezero.core_simple import using_config
    from dezero.core_simple import no_grad
    from dezero.core_simple import as_array
    from dezero.core_simple import as_variable
    from dezero.core_simple import setup_variable

else:
    from dezero.core import Variable
    from dezero.core import Function

setup_variable() # __init__.py에서 Variable 클래스에 대한 연산자 오버로딩도 다 해놓음!