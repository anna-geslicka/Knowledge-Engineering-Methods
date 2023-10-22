import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


def main():
    # dataset loading
    datasets = []
    my_path = "data"
    if os.path.exists(my_path):
        for i in range(1, 17):
            datasets.append(np.loadtxt(fname=f'data/dane{i}.txt'))

    for idx, dataset in enumerate(datasets):
        # train/test split
        print(f'Dataset {idx + 1}:')
        X = dataset[:, [0]]
        y = dataset[:, [1]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        print("Linear regression")
        # Fisher observation matrix
        l_observation = np.hstack([X_train, np.ones(X_train.shape)])
        # Moore-Penrose pseudo-inverse matrix * Y = identity matrix (vector)
        l_vector = np.linalg.pinv(l_observation) @ y_train
        l_w1 = l_vector[0][0]
        l_w0 = l_vector[1][0]
        print('Coefficient (w1):', l_w1, '\nIndependent term (w0):', l_w0)

        l_function = l_w1 * X_test + l_w0
        l_error = np.sqrt(np.mean((y_test - l_function) ** 2))
        print('LSE:', l_error)

        print("\nQuadrantic regression (x^4)")
        # Fisher observation matrix
        q_observation = np.hstack([X_train * X_train * X_train * X_train, X_train * X_train * X_train,
                                   X_train * X_train, X_train, np.ones(X_train.shape)])
        # Moore-Penrose pseudo-inverse matrix * Y = identity matrix (vector)
        q_vector = np.linalg.pinv(q_observation) @ y_train
        q_w4 = q_vector[0][0]
        q_w3 = q_vector[1][0]
        q_w2 = q_vector[2][0]
        q_w1 = q_vector[3][0]
        q_w0 = q_vector[4][0]
        print('1st coefficient (w4):', q_w4, '\n2nd coefficient (w3):', q_w3,
              '\n3rd coefficient (w2):', q_w1, '\n4th coefficient (w1):', q_w1, '\nIndependent term (w0):', q_w0)

        q_function = q_w4 * X_test ** 4 + q_w3 * X_test ** 3 + q_w2 * X_test ** 2 + q_w1 * X_test + q_w0
        q_error = np.sqrt(np.mean((y_test - q_function) ** 2))
        print('LSE:', q_error)

        print('\nLinear model better' if l_error < q_error else '\nQuadratic model better')

        print('---------------------')

        plt.plot(X, l_w1 * X + l_w0)
        plt.plot(X, q_w4 * X ** 4 + q_w3 * X ** 3 + q_w2 * X ** 2 + q_w1 * X + q_w0)
        plt.plot(X, y, 'ro')
        plt.show()


if __name__ == '__main__':
    main()
