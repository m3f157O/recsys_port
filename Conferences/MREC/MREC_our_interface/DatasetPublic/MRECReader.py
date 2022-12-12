import numpy as np
import scipy.sparse as sp
from pandas.api.types import CategoricalDtype
import os
import pandas as pd



"""
    Transform a dataset into a coo matrix
"""
def pandas_df_to_coo(dataset,scores=True):
    users = dataset["user_id"].unique()
    movies = dataset["item_id"].unique()
    shape = (len(users), len(movies))

    user_cat = CategoricalDtype(categories=sorted(users), ordered=True)
    movie_cat = CategoricalDtype(categories=sorted(movies), ordered=True)
    user_index = dataset["user_id"].astype(user_cat).cat.codes
    movie_index = dataset["item_id"].astype(movie_cat).cat.codes
    if scores:
        data = dataset.to_numpy()[:, 2]
    else:
        data_len = len(dataset["user_id"])
        data = np.ones(data_len)  #

    return sp.coo_matrix((data, (user_index, movie_index)), shape=shape)


"""
    Read the dataset and splt it into URM train and URM test
"""
def preprocessing_interactions_pandas(dataset, interactions,scores=True):

    count =0
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


    dataset = pandas_df_to_coo(dataset,scores=scores)
    URMtoPandasCsvWithScores("./Conferences/MREC/MREC_our_interface/dataset.txt", dataset)

    import matlab.engine

    eng = matlab.engine.start_matlab()
    matlab_script_directory = os.getcwd() + "/Conferences/MREC/MREC_our_interface/"
    eng.cd(matlab_script_directory)
    eng.prova(nargout=0)

    dataset_train = pd.read_csv("./Conferences/MREC/MREC_our_interface/train.txt", sep=' ',engine='python')
    dataset_train.columns = ['user_id', 'item_id', 'rating']

    ##max for no error (reindexing needed, not wanted)

    row=dataset_train.to_numpy()[:,0]
    col=dataset_train.to_numpy()[:,1]
    data=dataset_train.to_numpy()[:, 2]

    URM_train=sp.coo_matrix((data,(row,col)),shape=(int(max(row))+1, int(max(col))+1))


    dataset_test = pd.read_csv("./Conferences/MREC/MREC_our_interface/test.txt", sep=' ',engine='python')
    dataset_test.columns = ['user_id', 'item_id', 'rating']


    row=dataset_test.to_numpy()[:,0]
    col = dataset_test.to_numpy()[:,1]
    data = dataset_test.to_numpy()[:, 2]


    URM_test = sp.coo_matrix((data,(row,col)),shape=(int(max(row))+1, int(max(col))+1))

    return URM_train, URM_test

"""
    Write an URM into a file with the generic scores
"""
def URMtoPandasCsv(path, URM):
    column = (URM.col).copy()
    row = (URM.row).copy()
    data = (URM.data).copy()
    number_users = np.unique(row)
    file = open(path , "w")

    for i in range(len(number_users)):
        count = np.count_nonzero(row == i)
        items_to_add = column[:count]
        items = items_to_add
        column = column[count:]
        data = data[count:]
        for j in range(len(items)):
            # only users and items matters because the preprocessing is already done in the reader
            file.write(str(i + 1) + "\t" + str(items[j]) + "\t" + "1.0\n")


"""
    Write an URM into a file with the specific scores
"""
def URMtoPandasCsvWithScores(path, URM):
    column = (URM.col).copy()
    row = (URM.row).copy()
    number_users = np.unique(row)
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
            file.write(str(i) + "\t" + str(items[j]) + "\t" + str(int(datas[j]))+"\n")


"""
    Transform an adjency list into a coo matrix
"""
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


    return sp.coo_matrix((datas, (rows, cols)), shape=(len(lists), len(np.unique(cols))))





