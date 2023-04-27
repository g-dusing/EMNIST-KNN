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
    train_x = train_x.reshape(124800, 784)
    flat = []
    for i in train_x:
        flat.append(i.flatten())
    flat = np.array(flat)
    df = pd.DataFrame(flat)
    df["Label"] = train_y
    return df


if __name__ == "__main__":
    preprocess()
