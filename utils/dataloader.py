import numpy as np
import torch
import scipy.signal as signal
import os
import math


def load_dataset_mul(dataset_path, dataset_name, normalize_timeseries=False, verbose=True) -> (np.array, np.array):
    if verbose: print("Loading train / test dataset : ", dataset_name)

    root_path = dataset_path + '/' + dataset_name + '/'
    x_train_path = root_path + "X_train.npy"
    y_train_path = root_path + "y_train.npy"
    x_test_path = root_path + "X_test.npy"
    y_test_path = root_path + "y_test.npy"

    if os.path.exists(x_train_path):
        X_train = np.load(x_train_path).astype(np.float32)
        y_train = np.squeeze(np.load(y_train_path))
        X_test = np.load(x_test_path).astype(np.float32)
        y_test = np.squeeze(np.load(y_test_path))
    else:
        raise FileNotFoundError('File %s not found!' % (dataset_name))

    is_timeseries = True

    # extract labels Y and normalize to [0 - (MAX - 1)] range
    nb_classes = len(np.unique(y_train))
    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)

    if is_timeseries:
        # scale the values
        if normalize_timeseries:
            X_train_mean = X_train.mean()
            X_train_std = X_train.std()
            X_train = (X_train - X_train_mean) / (X_train_std + 1e-8)

    if verbose: print("Finished processing train dataset..")

    # extract labels Y and normalize to [0 - (MAX - 1)] range
    nb_classes = len(np.unique(y_test))
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)

    if is_timeseries:
        # scale the values
        if normalize_timeseries:
            X_test = (X_test - X_train_mean) / (X_train_std + 1e-8)

    if verbose:
        print("Finished loading test dataset..")
        print()
        print("Number of train samples : ", X_train.shape[0], "Number of test samples : ", X_test.shape[0])
        print("Number of classes : ", nb_classes)
        print("Sequence length : ", X_train.shape[-1])

    return X_train, y_train, X_test, y_test, nb_classes


def stft(x, **params):
    f, t, zxx = signal.stft(x, **params)
    return zxx


def stft_fun(input, dataset_name='lp4'):
    '''
    STFT transformation function
    '''
    k = input.shape[1]
    if dataset_name == 'lp4': len, fs = 30, 30
    elif dataset_name == 'lp5': len, fs = 30, 30

    for i in range(k):
        one_channel = input[:, i, :]
        if one_channel.shape[1] < len:
            n = math.ceil(len / one_channel.shape[1])
            input_new = np.empty((one_channel.shape[0], one_channel.shape[1] * n))
            for n_i in range(n):
                input_new[:, one_channel.shape[1] * n_i:one_channel.shape[1] * (n_i + 1)] = one_channel
        else:
            input_new = one_channel
        overlap = math.ceil(fs - input_new[:, :].shape[1] / len)
        length_new = (fs - overlap) * len
        temp = stft(input_new[:, :length_new], nperseg=fs, nfft=len*2-2, noverlap=overlap)
        temp = np.expand_dims(temp, axis=1)
        if i == 0:
            output = temp
        elif i != 0:
            output = np.concatenate((output, temp), axis=1)
    return torch.from_numpy(output[:, :, :, :])
