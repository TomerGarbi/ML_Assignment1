import numpy as np
from nearest_neighbour import learnknn, predictknn, gensmallm
import matplotlib.pyplot as plt


def Q2a():
    sample_sizes = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    k = 1
    data = np.load('mnist_all.npz')
    train2 = data["train2"]
    train3 = data["train3"]
    train5 = data["train5"]
    train6 = data["train6"]
    test2 = data["test2"]
    test3 = data["test3"]
    test5 = data["test5"]
    test6 = data["test6"]
    x_test = np.concatenate((test2, test3, test5, test6))
    y_test = [2 for _ in test2]
    y_test.extend([3 for _ in test3])
    y_test.extend([5 for _ in test5])
    y_test.extend([6 for _ in test6])
    max_err = []
    min_err = []
    avg_err = []
    for size in sample_sizes:
        result = []
        for i in range(10):
            x_train, y_train = gensmallm([train2, train3, train5, train6], [2, 3, 5, 6], size)
            classifier = learnknn(k, x_train, y_train)
            counter = 0
            predictions = predictknn(classifier, x_test)
            for i in range(len(y_test)):
                if predictions[i] == y_test[i]:
                    counter += 1
            result.append(counter / len(y_test))
        min_err.append(1 - max(result))
        max_err.append(1 - min(result))
        avg_err.append(1 - sum(result) / len(result))

    plt.plot(sample_sizes, avg_err, color="#ff0000", label="average error")
    plt.plot(sample_sizes, min_err, color='#00ff00', linestyle="--", label="minimum error")
    plt.plot(sample_sizes, max_err, color="#000000", linestyle="--", label="maximum error")

    plt.xlabel("Sample Size")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.ylim((0, 1))
    plt.show()


Q2a()



