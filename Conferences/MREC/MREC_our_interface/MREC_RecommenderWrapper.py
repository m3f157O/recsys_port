

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

        import numpy as np


        assert self.USER_factors.shape[0] == self.URM_train.shape[0]
        assert self.ITEM_factors.shape[0] == self.URM_train.shape[1]
            #todo fix conflict, add 1 to all users and items id when rewriting urm
        assert self.USER_factors.shape[1] == self.ITEM_factors.shape[1]

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

