import numpy as np
from dezero.models import MLP
from dezero.optimizers import SGD
import dezero.functions as F

# Dataset
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# Hyperparameter
lr = 0.2
iters = 10000
hidden_size = 10

# Model
model = MLP((10, 1))

# Optimizer -> 초기화하면서 타겟 지정(setup) 및 SGD 알고리즘에서 요구하는 learning rate 지정
optimizer = SGD(lr).setup(model)

for iter in range(iters):
    # Forward
    y_pred = model(x)
    loss = F.mean_squared_error(y_pred, y)

    # Grad Clear & Backward
    model.cleargrads()
    loss.backward()

    # Update with Optimizer
    optimizer.update()
    if iter % 1000 == 0:
        print(f"{iter + 1} loss : {loss.data}")