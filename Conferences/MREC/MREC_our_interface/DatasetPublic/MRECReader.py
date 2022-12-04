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
    data=np.ones(data_len)
    return sp.coo_matrix((data, (user_index, movie_index)), shape=shape)


def preprocessing_interactions_pandas(dataset, interactions,file,filename):

    # filtering users and items with less than 10 interactions
    filtered_users = dataset.groupby(["user_id"]).size().reset_index(name='interactions')
    filtered_users = filtered_users[filtered_users['interactions'] >= interactions]
    dataset = dataset[dataset["user_id"].isin(filtered_users["user_id"])]

    filtered_items = dataset.groupby(["item_id"]).size().reset_index(name='interactions')
    filtered_items = filtered_items[filtered_items['interactions'] >= interactions]
    dataset = dataset[dataset["item_id"].isin(filtered_items["item_id"])]


    dataset_train, dataset_test = skl.train_test_split(dataset, test_size=0.80, random_state = 42, shuffle = True)

    # writing training and testing into files
    dataset_train.to_csv(file+"train.txt", sep='\t', index=False, header=False)
    dataset_test.to_csv(file+"test.txt", sep='\t', index=False, header=False)
    URM_test = pandas_df_to_coo(dataset_test)
    URM_train = pandas_df_to_coo(dataset_train)


    return URM_train, URM_test

