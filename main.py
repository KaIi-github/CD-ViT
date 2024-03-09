import argparse
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import seaborn as sns

from models.CDViT import CDViT
from utils.modelUse import modelUse
from utils.dataloader import load_dataset_mul


torch.manual_seed(1234)


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-pre_train', type=str, default=False)
    parser.add_argument('-train', type=str, default=True)

    # Path parameters
    parser.add_argument('-MODEL_LOAD_PATH', type=str, default='')
    parser.add_argument('-RESULT_LOG_PATH', type=str, default='result_path')
    parser.add_argument('-UCI_DATASET_PATH', type=str, default=r'./datasets/UCI/')

    # Train parameters
    parser.add_argument('-BATCH_SIZE', type=str, default=16)
    parser.add_argument('-epochs', type=int, default=300)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-num_workers', type=int, default=0)

    # Model parameters
    parser.add_argument('-model', type=str, default='')
    parser.add_argument('-embed_dim', type=int, default=384, help="embedding dim")
    parser.add_argument('-f_size', type=int, default=128, help="input spectrogram frequence size")
    parser.add_argument('-t_size', type=int, default=128, help="input spectrogram time size")
    parser.add_argument('-patch_size', type=tuple, default=[8, 12, 16], help="patch size")
    parser.add_argument('-in_chans', type=int, default=1, help="channels size")
    parser.add_argument('-num_heads', type=int, default=12)
    parser.add_argument('-drop_rate', type=float, default=0.2)
    parser.add_argument('-d_output', type=int, default=2, help="Output length or num_classes")
    parser.add_argument('-depth', type=tuple, default=[4, 4, 4], help="depth of each layers")
    parser.add_argument('-fstride', type=tuple, default=[8, 12, 16], help="soft split freq stride, overlap=patch_size-stride")
    parser.add_argument('-tstride', type=tuple, default=[8, 12, 16], help="soft split time stride, overlap=patch_size-stride")
    parser.add_argument('-fshape', type=tuple, default=[8, 12, 16], help="shape of patch on the frequency dimension")
    parser.add_argument('-tshape', type=tuple, default=[8, 12, 16], help="shape of patch on the time dimension")

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    """ Main function. """
    opt = arg_parser()

    # Config
    sns.set()
    # default device is CUDA
    if torch.cuda.is_available():
        opt.device = torch.device('cuda')
        print(opt.device)
    else:
        opt.device = torch.device('cpu')
        print(opt.device)

    dataset_name_list = [
        'lp4', #  (75, 6, 15)
        'lp5', #  (64, 6, 15)
    ]
    for dataset_name in dataset_name_list:
        # Load dataset
        data_train, label_train, data_test, label_test, _ = load_dataset_mul(opt.UCI_DATASET_PATH, dataset_name=dataset_name)
        print('train data shape', data_train.shape)
        print('train label shape', label_train.shape)
        print('test data shape', data_test.shape)
        print('test label shape', label_test.shape)
        print('unique train label', np.unique(label_train))
        print('unique test label', np.unique(label_test))

        # covert numpy to pytorch tensor and put into gpu
        data_train = torch.from_numpy(data_train)
        data_train = data_train.to(opt.device)
        label_train = torch.from_numpy(label_train).int().to(opt.device)

        data_test_n = data_test
        label_test_n = label_test
        data_test = torch.from_numpy(data_test)
        data_test = data_test.to(opt.device)
        label_test = torch.from_numpy(label_test).int().to(opt.device)

        n_class = max(label_train) + 1
        opt.in_chans = data_train.shape[1]

        if dataset_name == 'lp4': img_size, patch_size = [32, 32, 32], [2, 3, 4]
        elif dataset_name == 'lp5': img_size, patch_size = [32, 32, 32], [2, 3, 4]

        CDViT= CDViT(img_size=img_size, in_chans=opt.in_chans, num_classes=n_class, num_heads=[opt.num_heads, opt.num_heads, opt.num_heads],
                                  patch_size=patch_size, embed_dim=[opt.embed_dim, opt.embed_dim, opt.embed_dim], depth=[[2, 2, 2, 0]],
                                  mlp_ratio=[4, 4, 4, 1], qkv_bias=True, clsFusion=False, selfornot=False,
                                  norm_layer=partial(nn.LayerNorm, eps=1e-6)).to(opt.device)
        # creat model and log save place,
        model_package = modelUse(opt=opt, dataset_name=dataset_name, model=CDViT)
        model_package.train(data_train, label_train, data_test, label_test)

