import numpy as np
import scipy.sparse as sp
import sklearn.model_selection as skl
from pandas.api.types import CategoricalDtype


import pandas as pd

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample


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


def preprocessing_interactions_pandas(dataset, interactions, file):

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


    dataset_train, dataset_test = skl.train_test_split(dataset, test_size=0.50, random_state = 42, shuffle = True)

    # writing training and testing into files

    URM_test = pandas_df_to_coo(dataset_test)
    URM_train = pandas_df_to_coo(dataset_train)
    URMtoPandasCsvWithScores(file+"train.txt",URM_train)
    URMtoPandasCsvWithScores(file+"test.txt",URM_test)

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
            file.write(str(i) + " " + str(items[j]) + " " + str(datas[j])+"\n")

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
    size = len(np.unique(rows))
    size2 = len(np.unique(cols))
    return sp.coo_matrix((datas, (rows, cols)), shape=(size2, size))





