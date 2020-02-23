from layer import *
from partition import *
import sys

alexnet = Sequential(
        Conv2d(64, [11, 11], stride = 4, padding = 2, input_size = [3, 224, 224]),
        Pool2d(64, [3, 3], 2),
        Conv2d(192, [5, 5], padding = 2),
        Pool2d(192, [3, 3], 2),
        Conv2d(384, [3, 3], padding = 1),
        Conv2d(256, [3, 3], padding = 1),
        Conv2d(256, [3, 3], padding = 1),
        Pool2d(256, [3, 3], 2),
        Flatten(),
        FC(4096),
        FC(4096),
        FC(16)
        )

optimized = optimize(alexnet, 30 if len(sys.argv) < 2 else int(sys.argv[1]))

print("Before optimization: ")
print(alexnet)
print()
print("After optimization: ")
print(optimized)
