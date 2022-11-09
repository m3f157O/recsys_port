import torch
from Conferences.IGN_CF.igcncf_github.utils import init_run

from dataset import get_dataset
from Conferences.IGN_CF.igcncf_github.config import get_gowalla_config, get_yelp_config, get_amazon_config
from tensorboardX import SummaryWriter
import os, sys
import numpy as np
import scipy.sparse as sp
import time

###THIS CODE IS FROM run.py FROM ORIGINAL IMPLEMENTATION
def init_file_and_device():
    log_path = __file__[:-3]
    init_run(log_path, 2021)
    device = torch.device('cpu')
    return device, log_path


def acquire_dataset(log_path, config):
    dataset_config, model_config, trainer_config = config[2]
    #dataset_config['path'] = dataset_config['path'][:-4] + str(1)
    writer = SummaryWriter(log_path)
    dataset = get_dataset(dataset_config)
    return dataset


def txt_to_csv(folder):
    for filename in os.listdir(folder):
        infilename = os.path.join(folder, filename)
        if not os.path.isfile(infilename): continue
        oldbase = os.path.splitext(filename)
        newname = infilename.replace('.txt', '.csv')
        os.rename(infilename, newname)


def adjacencyList2COO(toCOOarray):
    start = time.time()
    cols = np.array([])
    rows = np.array([])
    datas = np.array([])
    for index in range(len(toCOOarray)):
        adj = np.array(toCOOarray[index])  # get number of elements on given row
        data = np.ones_like(adj)  # all data is both 0 or 1
        row = np.ones_like(adj)
        row.fill(index)  # get correct row number
        col = adj  # useless
        datas = np.append(datas, data)  # concatenate to all
        rows = np.append(rows, row)
        cols = np.append(cols, col)
    end = time.time()
    print(end-start)
    return datas, rows, cols

def preprocessing(n_users, n_items,URM_val,URM_train,URM_test):

    items_train = (URM_train.col).copy()
    users_train = (URM_train.row).copy()
    items_val = (URM_val.col).copy()
    users_val = (URM_val.row).copy()
    items_test = (URM_test.col).copy()
    users_test = (URM_test.row).copy()
    for i in range(n_users):
        count_train = np.count_nonzero(users_train == i)
        count_val = np.count_nonzero(users_val == i)
        count_test = np.count_nonzero(users_test == i)

        if count_train + count_val + count_test < 10:
            np.delete(URM_train, i, axis=0)
            np.delete(URM_val, i, axis=0)
            np.delete(URM_test, i, axis=0)


    for i in range(n_items):
        count_train = np.count_nonzero(items_train == i)
        count_val = np.count_nonzero(items_val == i)
        count_test = np.count_nonzero(items_test == i)

        if count_train + count_val + count_test < 10:
            np.delete(URM_train, i, axis=1)
            np.delete(URM_val, i, axis=1)
            np.delete(URM_test, i, axis=1)

    return URM_val, URM_train, URM_test

