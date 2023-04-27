from emnist import extract_training_samples
from emnist import extract_test_samples
import numpy as np
import pandas as pd


def preprocess(s="letters"):
    '''
    Preprocess data taken from the emnist dataset
    :param s: String, can be any of the following: "balanced", "byclass", "bymerge", "digits", "letters", "mnist"
    :return: returns pandas dataframe containing data and corresponding lables
    '''
    train_x, train_y = extract_training_samples(s)
    test_x, test_y = extract_test_samples(s)
    # create training dataframe
    train_shape = train_x.shape
    train_x = train_x.reshape(train_shape[0], train_shape[1] * train_shape[2])
    flat_train = []
    for i in train_x:
        flat_train.append(i.flatten())
    flat_train = np.array(flat_train)
    df_train = pd.DataFrame(flat_train)
    df_train["Label"] = train_y
    # create testing dataframe
    test_shape = test_x.shape
    test_x = test_x.reshape(test_shape[0], test_shape[1] * test_shape[2])
    flat_test = []
    for j in test_x:
        flat_test.append(j.flatten())
    flat_test = np.array(flat_test)
    df_test = pd.DataFrame(flat_test)
    df_test["Label"] = test_y
    return df_train, df_test


if __name__ == "__main__":
    preprocess()
