import numpy as np
import scipy.sparse as sp
import sklearn.model_selection as skl
from pandas.api.types import CategoricalDtype


import pandas as pd

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

import matlab.engine


def pandas_df_to_coo(dataset):
    users = dataset["user_id"].unique()
    movies = dataset["item_id"].unique()
    shape = (len(users), len(movies))

    user_cat = CategoricalDtype(categories=sorted(users), ordered=True)
    movie_cat = CategoricalDtype(categories=sorted(movies), ordered=True)
    user_index = dataset["user_id"].astype(user_cat).cat.codes
    movie_index = dataset["item_id"].astype(movie_cat).cat.codes
    data_len=len(dataset["user_id"])
    data=np.ones(data_len) #todo fix
    return sp.coo_matrix((data, (user_index, movie_index)), shape=shape)


def preprocessing_interactions_pandas(dataset, interactions, file,n_users,n_items):

    # filtering users and items with less than 10 interactions
    while(True):
        n_users = len(dataset.groupby(["user_id"]).size())
        n_items = len(dataset.groupby(["item_id"]).size())

        filtered_users = dataset.groupby(["user_id"]).size().reset_index(name='interactions')
        filtered_users = filtered_users[filtered_users['interactions'] >= interactions]
        dataset = dataset[dataset["user_id"].isin(filtered_users["user_id"])]

        filtered_items = dataset.groupby(["item_id"]).size().reset_index(name='interactions')
        filtered_items = filtered_items[filtered_items['interactions'] >= interactions]
        dataset = dataset[dataset["item_id"].isin(filtered_items["item_id"])]

        n_users_updated = len(dataset.groupby(["user_id"]).size())
        n_items_updated = len(dataset.groupby(["item_id"]).size())

        if n_users == n_users_updated and n_items == n_items_updated:
            break


    dataset = pandas_df_to_coo(dataset)
    URMtoPandasCsvWithScores("./Conferences/MREC/MREC_our_interface/dataset.txt", dataset)

    import matlab.engine

    eng = matlab.engine.start_matlab()
    matlab_script_directory = os.getcwd() + "/Conferences/MREC/MREC_our_interface/"
    eng.cd(matlab_script_directory)
    eng.prova(nargout=0)

    ##with open('./Conferences/MREC/MREC_our_interface/train.txt', 'r') as input_file:
    ##    lines = input_file.readlines()
    ##    newLines = []
    ##    for line in lines:
    ##        newLine = line.strip(' ').split()
    ##        newLines.append(newLine)
    ##URM_train = preprocessing_interactions_lists(newLines)

    #àwith open('./Conferences/MREC/MREC_our_interface/test.txt', 'r') as input_file:
    ##    lines = input_file.readlines()
    ##    newLines = []
    ##        for line in lines:
    ##        newLine = line.strip(' ').split()
    ##        newLines.append(newLine)

    #URM_test = preprocessing_interactions_lists(newLines)
    dataset_train = pd.read_csv("./Conferences/MREC/MREC_our_interface/train.txt", sep=' ',engine='python')
    dataset_train.columns = ['user_id', 'item_id', 'rating']

    ##max for no error (reindexing needed, not wanted)

    row=dataset_train.to_numpy()[:,0]
    col=dataset_train.to_numpy()[:,1]
    data=dataset_train.to_numpy()[:, 2]
    print(max(row))
    print(max(col))

    URM_train=sp.coo_matrix((data,(row,col)),shape=(n_users+1, n_items+1))


    dataset_test = pd.read_csv("./Conferences/MREC/MREC_our_interface/test.txt", sep=' ',engine='python')
    dataset_test.columns = ['user_id', 'item_id', 'rating']
    print(len(np.unique(dataset_train.to_numpy()[:,1])))


    row=dataset_test.to_numpy()[:,0]
    col=dataset_test.to_numpy()[:,1]
    data=dataset_test.to_numpy()[:, 2]
    print(max(row))
    print(max(col))

    URM_test=sp.coo_matrix((data,(row,col)),shape=(n_users+1,n_items+1))

    return URM_train, URM_test

def URMtoPandasCsv(path,URM):
    column = (URM.col).copy()
    row = (URM.row).copy()
    data = (URM.data).copy()
    number_users = np.unique(row)
    number_items = np.unique(column)
    file = open(path , "w")

    for i in range(len(number_users)):
        count = np.count_nonzero(row == i)
        items_to_add = column[:count]
        items = items_to_add
        column = column[count:]
        data = data[count:]
        for j in range(len(items)):
            # only users and items matters because the preprocessing is already done in the reader
            file.write(str(i + 1) + "\t" + str(items[j]) + "\t" + "1.0")


def URMtoPandasCsvWithScores(path,URM):
    column = (URM.col).copy()
    row = (URM.row).copy()
    number_users = np.unique(row)
    number_items = np.unique(column)
    file = open(path , "w")
    data = (URM.data).copy()

    for i in range(len(number_users)):
        count = np.count_nonzero(row == i)
        items_to_add = column[:count]
        datas = data[:count]

        items = items_to_add
        column = column[count:]
        data = data[count:]
        for j in range(len(items)):
            # only users and items matters because the preprocessing is already done in the reader
            file.write(str(i+1) + "\t" + str(items[j]) + "\t" + str(int(datas[j]))+"\n")

def preprocessing_interactions_lists(lists):
    cols = np.array([])
    rows = np.array([])
    datas = np.array([])

    for index in range(len(lists)):
        adj = np.array(lists[index])  # get number of elements on given row
        data = np.ones_like(adj)  # all data is both 0 or 1
        row = np.ones_like(adj)
        row.fill(index)  # get correct row number
        col = adj  # useless
        datas = np.append(datas, data)  # concatenate to all
        rows = np.append(rows, row)
        cols = np.append(cols, col)
    n_row = len(rows)+1
    n_col = len(np.unique(cols))+1
    return sp.coo_matrix((datas, (rows, cols)), shape=(n_row, n_col))





