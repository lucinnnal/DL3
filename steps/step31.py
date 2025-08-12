import numpy as np
import dezero.functions as F
from dezero import Variable
from dezero.models import MLP

if __name__ == "__main__":
    model = MLP((10, 3))
    x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
    t = np.array([2, 0, 1, 0]) # label index
    y = model(x)

    loss = F.softmax_cross_entropy_simple(y, t)
    """
    def softmax_cross_entropy_simple(x, t):
        x, t = as_variable(x), as_variable(t)
        N = x.shape[0]
        p = softmax(x)
        p = clip(p, 1e-15, 1.0)  # To avoid log(0)
        log_p = log(p)
        tlog_p = log_p[np.arange(N), t.data] # Fancy Indexing => 2차원 배열에서 리스트(or array)를 넘겨주는 방식으로 인덱싱 하는 방법 -> 각 데이터 당 실제 정답에 대해 모델이 출력한 확률을 인덱싱

        y = -1 * sum(tlog_p) / N -> Batch Mean
        return y
    """

    print(loss)