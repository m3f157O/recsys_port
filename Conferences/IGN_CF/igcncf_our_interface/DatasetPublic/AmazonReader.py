#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/11/18

@author: Maurizio Ferrari Dacrema
"""
import numpy as np

from Recommenders.DataIO import DataIO
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
import scipy.sparse as sp
import os
from Data_manager.AmazonReviewData.AmazonBooksReader import AmazonBooksReader as AmazonBookReader_DataManager


class AmazonReader(object):
    URM_DICT = {}
    ICM_DICT = {}

    def __init__(self, pre_splitted_path):

        super(AmazonReader, self).__init__()

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data_"

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        dataIO = DataIO(pre_splitted_path)

        try:

            print("AmazonReader: Attempting to load pre-splitted data")

            for attrib_name, attrib_object in dataIO.load_data(pre_splitted_filename).items():
                self.__setattr__(attrib_name, attrib_object)


        except FileNotFoundError:

            print("AmazonReader: Pre-splitted data not found, building new one")

            print("AmazonReader: loading URM")

            # Done Replace this with the publicly available dataset you need
            #  The DataManagers are in the Data_Manager folder, if the dataset is already there use that data reader
            data_reader = AmazonBookReader_DataManager()
            dataset = data_reader.load_data()

            URM_all = dataset.get_URM_all()


            # TODO MUST BE CHECKED Apply data preprocessing if required (for example binarizing the data, removing users ...)
            # binarize the data (only keep ratings >= 3) and users and items must have more than 10 interactions
            URM_all.data = URM_all.data >= 3.0
            URM_all.eliminate_zeros()
            for user_id in range(np.shape(URM_all)[0]):
                start_pos = URM_all.tocsr.indptr[user_id]
                end_pos = URM_all.tocsr.indptr[user_id + 1]
                if sum(URM_all.tocsr.indices[start_pos:end_pos]) < 10:
                    URM_all.tocsr.indices[start_pos:end_pos] = 0
            URM_all.eliminate_zeros()
            for item_id in range(np.shape(URM_all)[0]):
                start_pos = URM_all.tocsc.indptr[item_id]
                end_pos = URM_all.tocsc.indptr[item_id + 1]
                if sum(URM_all.tocsc.indices[start_pos:end_pos]) < 10:
                    URM_all.tocsc.indices[start_pos:end_pos] = 0
            URM_all.eliminate_zeros()

            # Done select the data splitting that you need, almost certainly there already is a function that does the splitting
            #  in the way you need, if you are not sure, ask me via email
            # Split the data in train 70, validation 10 and test 20
            URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.7)
            URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.9)

            # Done get the sparse matrices in the correct dictionary with the correct name
            # TODO ICM_DICT and UCM_DICT can be empty if no ICMs or UCMs are required
            sp.csr_matrix(URM_test)
            sp.csr_matrix(URM_train)
            sp.csr_matrix(URM_validation)

            self.ICM_DICT = {}
            self.UCM_DICT = {}

            self.URM_DICT = {
                "URM_train": URM_train,
                "URM_test": URM_test,
                "URM_validation": URM_validation,
            }

            # You likely will not need to modify this part
            data_dict_to_save = {
                "ICM_DICT": self.ICM_DICT,
                "UCM_DICT": self.UCM_DICT,
                "URM_DICT": self.URM_DICT,
            }

            dataIO.save_data(pre_splitted_filename, data_dict_to_save=data_dict_to_save)

            print("AmazonBookReader: loading complete")
