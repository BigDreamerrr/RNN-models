import h5py
import numpy as np

def encode_hf5_file(path):
    hf = h5py.File(path, 'r')

    data_name = None
    for name in hf.keys():
        data_name = name

    data = hf[data_name][:]
    hf.close()
    
    return data

def get_data():
    train_data = encode_hf5_file(r"D:\Dataset\cse512springhw5\train_data.h5")
    train_labels = encode_hf5_file(r"D:\Dataset\cse512springhw5\train_label.h5")

    val_data = encode_hf5_file(r"D:\Dataset\cse512springhw5\val_data.h5")
    val_labels = encode_hf5_file(r"D:\Dataset\cse512springhw5\val_label.h5")

    test_data = encode_hf5_file(r"D:\Dataset\cse512springhw5\test_data.h5")

    return train_data, train_labels, val_data, val_labels, test_data