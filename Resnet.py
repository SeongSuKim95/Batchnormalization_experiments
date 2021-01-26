import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
stride = 2
num_blocks = 4

image = mpimg.imread("Lenna.tif")
print(image.shape)
plt.imshow(image)
plt.show()


strides = [stride] + [1] * (num_blocks - 1)

print(strides)

test = np.zeros((128,10)).astype(np.uint8)
for i in range(128):
    for j in range(10):
        test[i][j] = j

output = torch.tensor(test)
print(output)

_,predicted = torch.max(output,1)
print(predicted)
targets = torch.ones([128]) * 9
correct = predicted.eq(targets).sum().item()

print(correct)

a = torch.randn(1,2,3,4,5)
print(a)

steps = []
steps.extend([1,2,3,4])
print(steps)