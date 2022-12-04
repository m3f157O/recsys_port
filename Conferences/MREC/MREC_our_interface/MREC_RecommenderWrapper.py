

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



class MREC_RecommenderWrapper(BaseItemCBFRecommender, BaseMatrixFactorizationRecommender, BaseTempFolder):


    RECOMMENDER_NAME = "MREC_RecommenderWrapper"

    DEFAULT_GSL_LIB_FOLDER = '/usr/lib/x86_64-linux-gnu/'

    #todo take the dataset lists

    def __init__(self, URM_train, ICM_train):
        super(MREC_RecommenderWrapper, self).__init__(URM_train, ICM_train)

    def fit(self,
            batch_size = 128,
            para_lv=10,
            para_lu=1,
            para_ln=1e3,
            epoch_sdae=1000,
            epoch_dae=500,
            temp_file_folder = None,
            gsl_file_folder = None):


        self.temp_file_folder = self._get_unique_temp_folder(input_temp_file_folder=temp_file_folder)

        if gsl_file_folder is None:
            print("{}: Using default gsl folder '{}'".format(self.RECOMMENDER_NAME, self.DEFAULT_GSL_LIB_FOLDER))
            self.gsl_folder = self.DEFAULT_GSL_LIB_FOLDER
        else:
            print("{}: Using gsl folder '{}'".format(self.RECOMMENDER_NAME, gsl_file_folder))
            self.gsl_folder = gsl_file_folder




        # input_user_file = 'ctr-data/folder45/cf-train-1-users.dat'
        # input_item_file = 'ctr-data/folder45/cf-train-1-items.dat'

        print("MREC_RecommenderWrapper: Saving temporary data files for matlab use ... ")

        n_features = self.ICM_train.shape[1]

        content_file = self.temp_file_folder + "ICM.mat"
        scipy.io.savemat(content_file, {"X": self.ICM_train.toarray()}, appendmat=False)

        input_user_file = self.temp_file_folder + "cf-train-users.dat"
        self._save_dat_file_from_URM(self.URM_train, input_user_file)

        input_item_file = self.temp_file_folder + "cf-train-items.dat"
        self._save_dat_file_from_URM(self.URM_train.T, input_item_file)

        print("MREC_RecommenderWrapper: Saving temporary data files for matlab use ... done!")

        print("MREC_RecommenderWrapper: Calling matlab.engine ... ")

        eng = matlab.engine.start_matlab()

        matlab_script_directory = os.getcwd() + "/Conferences/MREC/MREC_github/"
        matlab_backward_path_prefix = "../../../../"
        eng.cd(matlab_script_directory)

        # para_pretrain refers to a preexisting trained model. Setting it to False in order to pretrain from scratch
        load_previous_pretrained_model = False

        eng.cdl_main_with_params(
            ##TODO parameters
                     matlab_backward_path_prefix + self.temp_file_folder,
                     self.gsl_folder,
                     matlab_backward_path_prefix + input_user_file,
                     matlab_backward_path_prefix + input_item_file,
                     matlab_backward_path_prefix + content_file,
                     para_lv,
                     para_lu,
                     para_ln,
                     epoch_sdae,
                     epoch_dae,
                     load_previous_pretrained_model,
                     batch_size,
                     n_features,
                     nargout=0,)


        print("MREC_RecommenderWrapper: Calling matlab.engine ... done!")

        os.remove(content_file)
        os.remove(input_user_file)
        os.remove(input_item_file)

        print("MREC_RecommenderWrapper: Loading trained model from temp matlab files ... ")
        self.USER_factors = genfromtxt(self.temp_file_folder + "final-U.dat", delimiter=' ')
        self.ITEM_factors = genfromtxt(self.temp_file_folder + "final-V.dat", delimiter=' ')

        assert self.USER_factors.shape[0] == self.URM_train.shape[0]
        assert self.ITEM_factors.shape[0] == self.URM_train.shape[1]

        assert self.USER_factors.shape[1] == self.ITEM_factors.shape[1]

        print("MREC_RecommenderWrapper: Loading trained model from temp matlab files ... done!")
        self._clean_temp_folder(temp_file_folder=self.temp_file_folder)



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