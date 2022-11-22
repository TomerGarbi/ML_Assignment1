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
    rounds = 10
    for size in sample_sizes:
        result = []
        for r in range(rounds):
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
    plt.title("Q2a")
    plt.xlabel("Sample Size")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.ylim((0, 1))
    plt.show()


def Q2e():
    sample_size = 200
    k_values = [i for i in range(1, 12)]
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
    rounds = 10
    avg_err = np.zeros(len(k_values))
    for r in range(rounds):
        print(r)
        x_train, y_train = gensmallm([train2, train3, train5, train6], [2, 3, 5, 6], sample_size)
        result = []
        for k in k_values:
            classifier = learnknn(k, x_train, y_train)
            counter = 0
            predictions = predictknn(classifier, x_test)
            for i in range(len(y_test)):
                if predictions[i] == y_test[i]:
                    counter += 1
            result.append(1 - counter / len(y_test))
        avg_err += np.array(result)
    avg_err = avg_err / rounds
    plt.plot(k_values, avg_err)
    plt.show()



def change_labels(y_train):
    m = len(y_train)
    indices = np.array([i for i in range(m)])
    np.random.shuffle(indices)
    indices_to_replace = indices[:int(0.15 * m)]
    labels = np.array([2, 3, 5, 6])
    for i in indices_to_replace:
        label = y_train[i]
        new_label = labels[0]
        while new_label == label:
            np.random.shuffle(labels)
            new_label = labels[0]
        y_train[i] = new_label



def Q2f():
    sample_size = 200
    k_values = [i for i in range(1, 12)]
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
    rounds = 10
    avg_err = np.zeros(len(k_values))
    for r in range(rounds):
        print(r)
        x_train, y_train = gensmallm([train2, train3, train5, train6], [2, 3, 5, 6], sample_size)
        change_labels(y_train)
        result = []
        for k in k_values:
            classifier = learnknn(k, x_train, y_train)
            counter = 0
            predictions = predictknn(classifier, x_test)
            for i in range(len(y_test)):
                if predictions[i] == y_test[i]:
                    counter += 1
            result.append(1 - counter / len(y_test))
        avg_err += np.array(result)
    avg_err = avg_err / rounds
    plt.plot(k_values, avg_err)
    plt.show()



if __name__ == "__main__":
    Q2a()




