

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/12/18
@author: Maurizio Ferrari Dacrema
"""

import os

from numpy import genfromtxt
import scipy.sparse as sps
import pandas as pd

from Recommenders.BaseCBFRecommender import BaseItemCBFRecommender
from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Recommenders.BaseTempFolder import BaseTempFolder

try:
    import matlab.engine
except ImportError:
    print("MREC_RecommenderWrapper: Unable to import Matlab engine. Fitting of a new model will not be possible")

import numpy as np
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


class MREC_RecommenderWrapper(BaseMatrixFactorizationRecommender, BaseTempFolder):


    RECOMMENDER_NAME = "MREC_RecommenderWrapper"

    DEFAULT_GSL_LIB_FOLDER = '/usr/lib/x86_64-linux-gnu/'


    def __init__(self, URM_train):
        super(MREC_RecommenderWrapper, self).__init__(URM_train)

    def fit(self,
            article_hyperparameters=None,
            batch_size = 128,
            para_lv=10,
            para_lu=1,
            para_ln=1e3,
            epoch_sdae=1000,
            epoch_dae=500,
            temp_file_folder = None,
            gsl_file_folder = None):



        if(article_hyperparameters is None):
            article_hyperparameters={
                'alpha': 30,
                'K': 20,
                'max_iter': 20,
            }
        alpha=article_hyperparameters['alpha']
        K=article_hyperparameters['K']
        max_iter=article_hyperparameters['max_iter']



        # input_user_file = 'ctr-data/folder45/cf-train-1-users.dat'
        # input_item_file = 'ctr-data/folder45/cf-train-1-items.dat'

        print("MREC_RecommenderWrapper: Saving temporary data files for matlab use ... ")


        #self._save_format(URM_to_save=self.URM_train,file_full_path="Conferences/MREC/MREC_github/test/dataset/train.txt")


        print("MREC_RecommenderWrapper: Saving temporary data files for matlab use ... done!")

        print("MREC_RecommenderWrapper: Calling matlab.engine ... ")


        self.URM_train= sps.coo_matrix(self.URM_train)

        ##We should re convert the URM in the data structure needed, but sometimes there's a bug, reindexing by 1 both the whole
        ## user or item columns. It is still not solid so it will not be included in the final project (its a missing requirement)
        ## its difficult because the matlab parsing function starts from zero index for users, zero for items,
        # while the splitting matlab function starts from one, zero for items, this causes problem if the matrix is loaded
        # directly in coo format, and not converted from csv to coo like in preprocessing_interaction_pandas
        ## we have been forced to use it and not write a custom one because the authors didn't give the information
        ## on which split setup they were going to use.
        #URMtoPandasCsvWithScores("./Conferences/MREC/MREC_our_interface/train.txt",self.URM_train)


        ##another way could be just doing a "mock" split, in which the train data gets split with 1,0,0 split into train data again,
        ##but with the small change of indexes needed by the matlab functions
        #eng = matlab.engine.start_matlab()
        #matlab_script_directory = os.getcwd() + "/Conferences/MREC/MREC_our_interface/"
        #eng.cd(matlab_script_directory)
        #eng.mock_split(nargout=0)


        # can't convert the urm here
        eng = matlab.engine.start_matlab()
        matlab_script_directory = os.getcwd() + "/Conferences/MREC/MREC_github/test"
        eng.cd(matlab_script_directory)
        eng.test_script(alpha,K,max_iter,nargout=0)

        # para_pretrain refers to a preexisting trained model. Setting it to False in order to pretrain from scratch
        load_previous_pretrained_model = False



        print("MREC_RecommenderWrapper: Calling matlab.engine ... done!")



        print("MREC_RecommenderWrapper: Loading trained model from temp matlab files ... ")
        self.USER_factors = genfromtxt("Conferences/MREC/MREC_github/test/P.txt", delimiter=',')
        print(self.USER_factors.shape)
        self.ITEM_factors = genfromtxt("Conferences/MREC/MREC_github/test/Q.txt", delimiter=',')



        assert self.USER_factors.shape[0] == self.URM_train.shape[0]
        assert self.ITEM_factors.shape[0] == self.URM_train.shape[1]

        assert self.USER_factors.shape[1] == self.ITEM_factors.shape[1]

        self.URM_train= sps.csr_matrix(self.URM_train)

        print("MREC_RecommenderWrapper: Loading trained model from temp matlab files ... done!")


    def _compute_item_score(self, user_id_array, items_to_compute = None):
        """
        USER_factors is n_users x n_factors
        ITEM_factors is n_items x n_factors

        The prediction for cold users will always be -inf for ALL items

        :param user_id_array:
        :param items_to_compute:
        :return:
        """
        import numpy as np
        assert self.USER_factors.shape[1] == self.ITEM_factors.shape[1], \
            "{}: User and Item factors have inconsistent shape".format(self.RECOMMENDER_NAME)

        assert self.USER_factors.shape[0] > np.max(user_id_array),\
                "{}: Cold users not allowed. Users in trained model are {}, requested prediction for users up to {}".format(
                self.RECOMMENDER_NAME, self.USER_factors.shape[0], np.max(user_id_array))

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.n_items), dtype=np.float32)*np.inf
            item_scores[:, items_to_compute] = np.dot(self.USER_factors[user_id_array], self.ITEM_factors[items_to_compute,:].T)

        else:
            item_scores = np.dot(self.USER_factors[user_id_array], self.ITEM_factors.T)


        # No need to select only the specific negative items or warm users because the -inf score will not change
        if self.use_bias:
            item_scores += self.ITEM_bias + self.GLOBAL_bias
            item_scores = (item_scores.T + self.USER_bias[user_id_array]).T

        return item_scores

    def _save_dat_file_from_URM(self, URM_to_save, file_full_path):

        file_object = open(file_full_path, "w")

        URM_to_save = sps.csr_matrix(URM_to_save)


        n_rows, n_cols = URM_to_save.shape

        for row_index in range(n_rows):

            start_pos = URM_to_save.indptr[row_index]
            end_pos = URM_to_save.indptr[row_index +1]

            profile = URM_to_save.indices[start_pos:end_pos]

            new_line = "{} {}\n".format(len(profile), " ".join(str(element) for element in profile))

            file_object.write(new_line)

        file_object.close()

    def _save_format(self, URM_to_save, file_full_path):

        dataset = pd.DataFrame.sparse.from_spmatrix(URM_to_save)
        dataset.to_csv(file_full_path, sep='\t', index=False, header=False)

