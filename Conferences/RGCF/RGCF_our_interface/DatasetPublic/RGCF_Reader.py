import numpy as np
import scipy.sparse as sp


""""
    Method to make the preprocessing: checks if the number of interactions is greater than 15 
"""
def preprocessing(n_users, n_items,URM_val,URM_train,URM_test):
    URM_val_csr = URM_val.tocsr()
    URM_train_csr = URM_train.tocsr()
    URM_test_csr = URM_test.tocsr()

    for user_id in range(n_users):
        interactions_val = URM_val_csr.getrow(user_id).count_nonzero()
        interactions_train = URM_train_csr.getrow(user_id).count_nonzero()
        interactions_test = URM_test_csr.getrow(user_id).count_nonzero()
        if (interactions_val + interactions_train + interactions_test) < 15:
            # removing the row with user who has less than 10 interactions
            URM_train_csr = delete_row_csr(URM_train_csr,user_id)
            URM_val_csr = delete_row_csr(URM_val_csr, user_id)
            URM_test_csr = delete_row_csr(URM_test_csr, user_id)

    URM_val_csc = sp.csc_matrix(URM_val_csr)
    URM_train_csc = sp.csc_matrix(URM_train_csr)
    URM_test_csc = sp.csc_matrix(URM_test_csr)

    for item_id in range(n_items):
        interactions_val = URM_val_csc.getcol(item_id).count_nonzero()
        interactions_train = URM_train_csc.getcol(item_id).count_nonzero()
        interactions_test = URM_test_csc.getcol(item_id).count_nonzero()
        if (interactions_val + interactions_train + interactions_test) < 15:
            np.delete(URM_val_csc, item_id, 0)
            np.delete(URM_train_csc, item_id, 0)
            np.delete(URM_test_csc, item_id, 0)
    # matrix must be updated
    URM_val = sp.coo_matrix(URM_val_csc)
    URM_train = sp.coo_matrix(URM_train_csc)
    URM_test = sp.coo_matrix(URM_test_csc)
    return URM_val, URM_train, URM_test


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
