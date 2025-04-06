import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

observation = 1000
xs =np.random.uniform(low=-10, high=10, size=(observation, 1))
zs =np.random.uniform(-10, 10,(observation, 1))

inputs = np.column_stack((xs, zs))

print(inputs.shape)

noise = np.random.uniform(-1, 1,(observation,1))
targets = 2*xs + 3*zs +5 +noise
print(targets.shape)

targets = targets.reshape(observation)
fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(xs, zs, targets)
# ax.set_xlabel('xs')
# ax.set_ylabel('zs')
# ax.set_zlabel('Targets')
# ax.view_init(azim=100)
# plt.show()
targets = targets.reshape(observation, 1)
init_range = 0.1

weights = np.random.uniform(-init_range, init_range, size=(2, 1))
baises = np.random.uniform(-init_range, init_range, size=1)
learning_rate = 0.02

for i in range(200):
    output = np.dot(inputs, weights) + baises
    deltas = output - targets
    loss = np.sum(deltas**2)/ 2 / observation

    print('Loss',loss)

    deltas_scaled = deltas/observation
    weights = weights -learning_rate * np.dot(inputs.T, deltas_scaled)
    baises = baises - learning_rate * np.sum(deltas_scaled)

plt.plot(output, targets)
plt.xlabel('outputs')
plt.ylabel('tagtes')
plt.show()