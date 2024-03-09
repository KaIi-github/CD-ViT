import os
from sklearn.metrics import accuracy_score
import numpy as np

from .dataloader import stft_fun


def eval_condition(iepoch,print_result_every_x_epoch):
    if (iepoch + 1) % print_result_every_x_epoch == 0:
        return True
    else:
        return False


def eval_model(model, dataloader):
    predict_list = np.array([])
    label_list = np.array([])
    for sample in dataloader:
        data = stft_fun(sample[0].squeeze(1).cpu()).float().to('cuda:0')
        y_predict = model(data).float()
        y_predict = y_predict.detach().cpu().numpy()
        y_predict = np.argmax(y_predict, axis=1)
        predict_list = np.concatenate((predict_list, y_predict), axis=0)
        label_list = np.concatenate((label_list, sample[1].long().detach().cpu().numpy()), axis=0)
    acc = accuracy_score(predict_list, label_list)
    return acc


def save_to_log(sentence, Result_log_folder, dataset_name):
    father_path = Result_log_folder + dataset_name
    if not os.path.exists(father_path):
        os.makedirs(father_path)
    path = father_path + '/' + dataset_name + '_.txt'
    with open(path, "a") as myfile:
        myfile.write(sentence + '\n')
