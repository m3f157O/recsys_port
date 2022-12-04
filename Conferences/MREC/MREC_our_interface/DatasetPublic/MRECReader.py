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





    dataset.to_csv(file+"processed_"+filename, sep='\t', index=False)



    return URM_train, URM_test

