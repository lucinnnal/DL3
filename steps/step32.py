import math
import numpy as np
import dezero
import dezero.functions as F
import dezero.optimizers as optim
from dezero.datasets import get_spiral
from dezero.models import MLP



if __name__ == "__main__":
    # Hyperparameters
    max_epoch = 30
    hidden_size = 10
    batch_size = 30
    lr = 1.0

    # Dataset & Optimizer & Model
    x, t = get_spiral(train=True)
    model = MLP((hidden_size, 3))
    optimizer = optim.SGD(lr=lr).setup(model)
    
    # 최대 iteration 구하기
    data_size = len(x)
    max_iter = math.ceil(data_size / batch_size)

    for epoch in range(max_epoch):
        # Dataset Index Shuffling
        index = np.random.permutation(data_size)
        sum_loss = 0

        for i in range(max_iter):
            # Mini-Batch
            batch_index = index[i * batch_size:(i+1) * batch_size] # permutation된 batch_index list에서 매 iteration 마다 batch 크기 만큼의 index들을 뽑아오기
            # data로부터 batch 추출
            x_batch = x[batch_index]
            t_batch = t[batch_index]

            # Forward
            y = model(x_batch)
            loss = F.softmax_cross_entropy(y, t_batch)

            # Cleargrads
            model.cleargrads()

            # Backward
            loss.backward()

            # Update
            optimizer.update()

            # Loss calc
            sum_loss += float(loss.data) * len(t_batch)

        avg_loss = sum_loss / data_size
        print(f"epoch : {epoch + 1}, loss : {avg_loss}") 