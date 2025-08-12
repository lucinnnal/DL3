class Optimizer:
    def __init__(self):
        self.target = None # update할 매개변수를 containing하고 있는 클래스 저장
        self.hooks = [] # 매개변수 업데이트 전에 필요한 전처리를 담당하는 메서드 모임 (lr rate scheduling, weight decay, gradient clipping 등)
    
    def setup(self, target):
        self.target = target
        return self
    
    def update(self):
        # 1.Gradient가 None이 아닌 Params들을 리스트에 모아두기
        params = [p for p in self.target.params() if p.grad is not None]

        # 2.Preprocessing Parameters before update (f are preprocessing functions)
        for f in self.hooks:
            f(params)
        
        # 3.Update
        for param in params:
            self.update_one(param)

    # 실질적으로 개별 parameter들에 업데이트를 적용하는 메서드 (Optimizer 종류에 따라 다르게 오버라이딩)   
    def update_one(self, param):
        raise NotImplementedError()

    # 매개변수 전처리 함수 추가 (Optimizer 종류에 따라 다르게 오버라이딩)   
    def add_hook(self, f):
        self.hooks.append(f)

class SGD(Optimizer):
    def __init__(self, lr=0.1):
        super().__init__()
        self.lr = lr
    
    def update_one(self, param):
        param.data -= self.lr * param.grad.data