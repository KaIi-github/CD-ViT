import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from .log_manager import eval_condition, eval_model, save_to_log
from .dataloader import stft_fun


class modelUse:
    def __init__(self, opt, dataset_name, model):
        super(modelUse, self).__init__()

        # Define logs and save training content
        if not os.path.exists(opt.RESULT_LOG_PATH + dataset_name + '/'):
            os.makedirs(opt.RESULT_LOG_PATH + dataset_name + '/')
        initial_model_path = opt.RESULT_LOG_PATH + dataset_name + '/' + dataset_name + 'initial_model'
        best_model_path = opt.RESULT_LOG_PATH + dataset_name + '/' + dataset_name + 'Best_model'


        self.opt = opt
        self.RESULT_LOG_PATH_folder = opt.RESULT_LOG_PATH
        self.dataset_name = dataset_name
        self.Best_model_path = best_model_path
        self.Initial_model_path = initial_model_path
        self.device = opt.device

        self.max_epoch = opt.epochs
        self.batch_size = opt.BATCH_SIZE
        self.print_result_every_x_epoch = 1
        self.lr = opt.lr
        self.model = model
        self.torch_CDViT = None

    def train(self, data_train, label_train, data_test, label_test):
        print('code is running on ', self.device)

        CDViT = self.model

        # loss, optimizer, scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(CDViT.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, min_lr=0.0001)
        """ number of parameters """
        num_params = sum(p.numel() for p in CDViT.parameters() if p.requires_grad) / 1_000_000
        print('[Info] Number of parameters: %.3fM' % num_params)

        # build dataloader
        train_dataset = TensorDataset(data_train, label_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.opt.num_workers)
        test_dataset = TensorDataset(data_test, label_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                 num_workers=self.opt.num_workers)

        CDViT.train()
        temp_acc = 0
        for i in range(self.max_epoch):
            with tqdm(total=len(train_loader.dataset), desc=f"[Epoch {i + 1:3d}/{self.max_epoch}]") as pbar:
                for sample in train_loader:
                    optimizer.zero_grad()
                    data = stft_fun(sample[0].squeeze(1).cpu(), dataset_name=self.dataset_name).float().to(self.device)
                    y_predict = CDViT(data)
                    output = criterion(y_predict, sample[1].long())
                    output.backward(retain_graph=True)
                    optimizer.step()
                    pbar.update(sample[0].shape[0])
                scheduler.step(output)

                if eval_condition(i, self.print_result_every_x_epoch):
                    CDViT.eval()
                    acc_train = eval_model(CDViT, train_loader)
                    acc_test = eval_model(CDViT, test_loader)
                    CDViT.train()

                    sentence = 'train_acc=\t' + str(acc_train) + '\t test_acc=\t' + str(acc_test)
                    save_to_log(sentence, self.RESULT_LOG_PATH_folder, self.dataset_name)
                    if acc_test >= temp_acc:
                        torch.save(CDViT.state_dict(), self.Best_model_path)
                        temp_acc = acc_test
                pbar.set_postfix({'loss': '{:6f}'.format(output.item()),
                                  'train_acc': '{:6f}'.format(acc_train),
                                  'test_acc': '{:6f}'.format(acc_test)})
        self.torch_CDViT = CDViT

    def predict(self, data_test):
        data_test = torch.from_numpy(data_test)
        data_test.requires_grad = False
        data_test = data_test.to(self.device)

        if len(data_test.shape) == 2:
            data_test = data_test.unsqueeze_(1)

        test_dataset = TensorDataset(data_test)
        test_loader = DataLoader(test_dataset, batch_size=max(int(min(data_test.shape[0] / 10, self.batch_size)), 2),
                                 shuffle=False)

        self.torch_CDViT.eval()
        predict_list = np.array([])
        for sample in test_loader:
            data = stft_fun(sample[0].squeeze(1).cpu(), dataset_name=self.dataset_name).float().to(self.device)
            y_predict = self.torch_CDViT(data).float()
            y_predict = y_predict.detach().cpu().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)

        return predict_list
