#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/11/18

@author: Maurizio Ferrari Dacrema
"""

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

from Recommenders.DataIO import DataIO

from IGN_CFReader import *

from Conferences.IGN_CF.igcncf_github.config import get_gowalla_config, get_yelp_config, get_amazon_config

import numpy as np
import scipy.sparse as sp

import os
from Data_manager.AmazonReviewData.AmazonBooksReader import AmazonBooksReader as AmazonReader_DataManager


class AmazonReader(object):
    URM_DICT = {}
    ICM_DICT = {}

    def __init__(self, pre_splitted_path):

        super(AmazonReader, self).__init__()

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data_"

        dataIO = DataIO(pre_splitted_path)

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        device, log_path = init_file_and_device()

        config = get_gowalla_config(device)

        dataset = acquire_dataset(log_path, config)

        ##TODO PUT ^ IN EXCEPTION, CORRECTLY DOWNLOAD DATA IF NOT THERE ;) (OvO)
        try:

            print("AmazonReader: Attempting to load pre-splitted data")

            for attrib_name, attrib_object in dataIO.load_data(pre_splitted_filename).items():
                self.__setattr__(attrib_name, attrib_object)


        except FileNotFoundError:

            print("AmazonReader: Pre-splitted data not found, building new one")

            print("AmazonReader: loading URM")

            # DO Replace this with the publicly available dataset you need
            #  The DataManagers are in the Data_Manager folder, if the dataset is already there use that data reader
            data_reader = AmazonReader_DataManager()
            dataset = data_reader.load_data()

            URM_all = dataset.get_URM_all()

            # TODO Apply data preprocessing if required (for example binarizing the data, removing users ...)
            # binarize the data (only keep ratings >= 4)
            URM_all.data = URM_all.data >= 4.0
            URM_all.eliminate_zeros()
            # TODO MUST BE CHECKED Apply data preprocessing if required (for example binarizing the data, removing users ...)
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

            # TODO select the data splitting that you need, almost certainly there already is a function that does the splitting
            #  in the way you need, if you are not sure, ask me via email
            # Split the data in train, validation and test
            URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.8)
            URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.8)

            # Done get the sparse matrices in the correct dictionary with the correct name
            # TODO ICM_DICT and UCM_DICT can be empty if no ICMs or UCMs are required
            sp.csr_matrix(URM_test)
            sp.csr_matrix(URM_train)
            sp.csr_matrix(URM_validation)

            self.ICM_DICT = {}
            self.UCM_DICT = {}

            self.URM_DICT = {
                "URM_train": dataset.get_URM_all(),
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

            print("AmazonReader: loading complete")
