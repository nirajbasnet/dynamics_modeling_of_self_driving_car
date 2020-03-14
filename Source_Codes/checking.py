# Created by Aashish Adhikari at 9:56 PM 3/11/2020

import numpy as np

L = 2
N = 5
A = np.random.rand(L,L)

print(A)
print("-----------------")

print(np.tile(A,(N, 1, 1)))

print("-----------------")
print(np.tile(A,(N, 2, 2)))
