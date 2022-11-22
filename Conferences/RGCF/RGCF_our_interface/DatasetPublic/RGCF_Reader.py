import numpy as np



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

