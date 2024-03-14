import numpy as np
import os
from KNN import KNN
import time

# split the data into train and test in ratio 1:1
def t_t_split(data: np.ndarray) -> (np.ndarray, np.ndarray): # type: ignore
    permutationed_data = np.random.permutation(data)
    train_data, test_data = np.split(permutationed_data, 2)
    return train_data, test_data

def validation(data: np.ndarray, num_of_iter: int, k: int, distance_lp: float = 2):
    empirical_error = 0
    true_error = 0
    i = 0  # Counter variable
    while i < num_of_iter:
        classfier = KNN(k=k, distance_lp=distance_lp)
        # split the data into train and test in ratio 1:1
        train_data, test_data = t_t_split(data)
        # train the model
        classfier.train_model(train_data)
        # predict the labels of the test data
        y_pred_train = classfier.predict_labels(train_data[:, :-1]).astype(int)
        y_pred_test = classfier.predict_labels(test_data[:, :-1]).astype(int)
        # calculate the error of the model on the train and test data
        empirical_error += np.sum(y_pred_train != train_data[:, -1].astype(int)) / len(train_data)
        true_error += np.sum(y_pred_test != test_data[:, -1].astype(int)) / len(test_data)
        i += 1  # Increment counter

    # print the average error of the model on the train and test data
    print(f"KNN with k = {k} and Lp norm = {distance_lp}: empirical error: {empirical_error / num_of_iter}"
          f", true error: {true_error / num_of_iter}")


if __name__ == '__main__':
    start_time = time.time()
    
    random_seed = 82
    np.random.seed(random_seed)
    cwd = os.getcwd()
    path_to_data = os.path.join(cwd, "haberman.data")
    print("Haberman data set:")
    data = np.genfromtxt(path_to_data, delimiter=",", dtype=int)
    
    k_values = {1, 3, 5, 7, 9}
    p_values = {1, 2, np.inf}
    
    k_iterator = iter(k_values)
    p_iterator = iter(p_values)
    
    while True:
        try:
            k = next(k_iterator)
        except StopIteration:
            break
        p_iterator = iter(p_values)  # reset p_iterator for each k
        while True:
            try:
                p = next(p_iterator)
            except StopIteration:
                break
            validation(data, 100, k, p)

    print("===========================================")
    
    print("Circle Separator data set:")
    data = np.loadtxt("circle_separator.txt")
    # replace the labels from {-1, 1} to {0, 1}
    data[:, -1] = (data[:, -1] + 1) / 2
    
    k_iterator = iter(k_values)
    p_iterator = iter(p_values)
    
    while True:
        try:
            k = next(k_iterator)
        except StopIteration:
            break
        p_iterator = iter(p_values)  # reset p_iterator for each k
        while True:
            try:
                p = next(p_iterator)
            except StopIteration:
                break
            validation(data, 100, k, p)
            
    print(f"--- {time.time() - start_time} seconds ---")



