import numpy as np
import h5py

def load_train_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_x = np.array(train_dataset["train_set_x"][:]) # train images
    train_y = np.array(train_dataset["train_set_y"][:]) # train labels
    return train_x, train_y
    
def load_test_data():
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_x = np.array(test_dataset["test_set_x"][:]) # test images
    test_y = np.array(test_dataset["test_set_y"][:]) # test labels
    return test_x, test_y

def flatten(z):
    m, h, w, c = z.shape
    nx = h*w*c
    z = z.reshape((m, nx))
    return z.T # 'T' is for transpose