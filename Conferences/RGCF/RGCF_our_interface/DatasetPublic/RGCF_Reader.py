import numpy as np
import scipy.sparse as sp


""""
    Method to make the preprocessing: checks if the number of interactions is greater than 15 
"""
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

        if count_train + count_val + count_test < 15:
            np.delete(URM_train, i, axis=0)
            np.delete(URM_val, i, axis=0)
            np.delete(URM_test, i, axis=0)


    for i in range(n_items):
        count_train = np.count_nonzero(items_train == i)
        count_val = np.count_nonzero(items_val == i)
        count_test = np.count_nonzero(items_test == i)

        if count_train + count_val + count_test < 15:
            np.delete(URM_train, i, axis=1)
            np.delete(URM_val, i, axis=1)
            np.delete(URM_test, i, axis=1)

    return URM_val, URM_train, URM_test





def delete_row_csr(mat, i):

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
