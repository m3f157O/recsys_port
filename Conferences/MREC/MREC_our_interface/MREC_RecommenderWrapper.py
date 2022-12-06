

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

    #todo take the dataset lists

    def __init__(self, URM_train):
        super(MREC_RecommenderWrapper, self).__init__(URM_train)

    def fit(self,
            batch_size = 128,
            para_lv=10,
            para_lu=1,
            para_ln=1e3,
            epoch_sdae=1000,
            epoch_dae=500,
            temp_file_folder = None,
            gsl_file_folder = None):






        # input_user_file = 'ctr-data/folder45/cf-train-1-users.dat'
        # input_item_file = 'ctr-data/folder45/cf-train-1-items.dat'

        print("MREC_RecommenderWrapper: Saving temporary data files for matlab use ... ")


        #self._save_format(URM_to_save=self.URM_train,file_full_path="Conferences/MREC/MREC_github/test/dataset/train.txt")


        print("MREC_RecommenderWrapper: Saving temporary data files for matlab use ... done!")

        print("MREC_RecommenderWrapper: Calling matlab.engine ... ")

        eng = matlab.engine.start_matlab()
        matlab_script_directory = os.getcwd() + "/Conferences/MREC/MREC_github/test"
        eng.cd(matlab_script_directory)

        eng.test_script(nargout=0)

        # para_pretrain refers to a preexisting trained model. Setting it to False in order to pretrain from scratch
        load_previous_pretrained_model = False



        print("MREC_RecommenderWrapper: Calling matlab.engine ... done!")



        print("MREC_RecommenderWrapper: Loading trained model from temp matlab files ... ")
        self.USER_factors = genfromtxt("Conferences/MREC/MREC_github/test/P.txt", delimiter=',')
        print(self.USER_factors.shape)
        self.ITEM_factors = genfromtxt("Conferences/MREC/MREC_github/test/Q.txt", delimiter=',')
        print(self.ITEM_factors.shape)


        assert self.USER_factors.shape[0] == self.URM_train.shape[0]
        assert self.ITEM_factors.shape[0] == self.URM_train.shape[1]

        assert self.USER_factors.shape[1] == self.ITEM_factors.shape[1]

        print("MREC_RecommenderWrapper: Loading trained model from temp matlab files ... done!")



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

