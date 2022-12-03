import numpy as np
import scipy.sparse as sp
import pandas as pd
from pandas.api.types import CategoricalDtype



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


def preprocessing_interactions_pandas(dataset, interactions):


    filtered_users = dataset.groupby(["user_id"]).size().reset_index(name='interactions')
    filtered_users = filtered_users[filtered_users['interactions'] >= interactions]
    dataset = dataset[dataset["user_id"].isin(filtered_users["user_id"])]



    filtered_items = dataset.groupby(["item_id"]).size().reset_index(name='interactions')
    filtered_items = filtered_items[filtered_items['interactions'] >= interactions]
    dataset = dataset[dataset["item_id"].isin(filtered_items["item_id"])]








    #dataset.to_csv(file+"processed_"+filename, sep='\t', index=False)


    coo = pandas_df_to_coo(dataset)
    return coo




def preprocessing_interactions(n_users, n_items, URM_train, URM_test):
    URM_train_csr = URM_train.tocsr()
    URM_test_csr = URM_test.tocsr()

    for user_id in range(n_users):
        interactions_train = URM_train_csr.getrow(user_id).count_nonzero()
        interactions_test = URM_test_csr.getrow(user_id).count_nonzero()
        if (interactions_train + interactions_test) < 10:
            # removing the row with user who has less than 10 interactions
            URM_train_csr = delete_row(URM_train_csr, user_id)
            URM_test_csr = delete_row(URM_test_csr, user_id)

    URM_train_csc = sp.csc_matrix(URM_train_csr)
    URM_test_csc = sp.csc_matrix(URM_test_csr)

    for item_id in range(n_items):
        interactions_train = URM_train_csc.getrow(item_id).count_nonzero()
        interactions_test = URM_test_csc.getrow(item_id).count_nonzero()
        if ( interactions_train + interactions_test) < 10:
            URM_train_csc= delete_row(URM_train_csc, item_id)
            URM_test_csc = delete_row(URM_test_csc, item_id)
    # matrix must be updated
    URM_train = sp.coo_matrix(URM_train_csc)
    URM_test = sp.coo_matrix(URM_test_csc)

    return URM_train, URM_test





def delete_row(mat, i):

    n = mat.indptr[i+1] - mat.indptr[i]
    if n > 0:
        mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i+1]:]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i+1]:]
        mat.indices = mat.indices[:-n]
    mat.indptr[i:-1] = mat.indptr[i+1:]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0]-1, mat._shape[1])
    return mat
