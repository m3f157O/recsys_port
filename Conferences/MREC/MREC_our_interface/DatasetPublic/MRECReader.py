import scipy.sparse as sp
import pandas as pd

def preprocessing_ratings(file, interactions,filename):

    dataset = pd.read_csv(file+filename, sep='::')

    dataset.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    filtered = dataset.groupby(["user_id"]).size().reset_index(name='interactions')

    filtered = filtered[filtered['interactions'] >= interactions]

    filtered.to_csv(file+filename+"filtered", sep='\t', index=False)
    print(dataset["user_id"])
    print(filtered["user_id"])
    print(dataset["user_id"].isin(filtered["user_id"]))

    dataset = dataset[dataset["user_id"].isin(filtered["user_id"])]

    dataset.to_csv(file+"processed_"+filename, sep='\t', index=False)
    URM_all = dataset.astype(pd.SparseDtype("float64",0)).sparse.to_coo()

    return URM_all




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
