import numpy as np
from nearest_neighbour import learnknn, predictknn, gensmallm

def Q2a():
    sample_sizes = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    k = 1
    data = np.load('mnist_all.npz')
    for size in sample_sizes:
        (X, y) = gensmallm([labelAsample, labelBsample], [A, B], size)



(X, y) = gensmallm([labelAsample,labelBsample],[A, B], 10)