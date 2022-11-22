#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/11/18

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.DataIO import DataIO
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
import os
from recbole.data import create_dataset, data_preparation
from Data_manager.Movielens.Movielens20MReader import Movielens20MReader as Movielens20MReader_DataManager
from Conferences.RGCF.RGCF_our_interface.DatasetPublic.RGCF_Reader import preprocessing


class Movielens1MReader(object):

    URM_DICT = {}
    ICM_DICT = {}

    def __init__(self, pre_splitted_path,config):

        super(Movielens1MReader, self).__init__()

        #pre_splitted_path += "data_split/"
        pre_splitted_filename = "ml-1m-processed"

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        dataIO = DataIO(pre_splitted_path)

        try:

            print("Movielens20MReader: Attempting to load pre-splitted data")

            for attrib_name, attrib_object in dataIO.load_data(pre_splitted_filename).items():
                 self.__setattr__(attrib_name, attrib_object)


        except FileNotFoundError:

            print("Movielens20MReader: Pre-splitted data not found, building new one")

            print("Movielens20MReader: loading URM")

            dataset = create_dataset(config)
            train_data, valid_data, test_data = data_preparation(config, dataset)

            URM_train = train_data.dataset.inter_matrix(form='coo')
            URM_validation = valid_data.dataset.inter_matrix(form='coo')
            URM_test = test_data.dataset.inter_matrix(form='coo')

            n_users = URM_train.shape[0]
            n_items = URM_train.shape[1]
            URM_val, URM_train, URM_test = preprocessing(n_users, n_items, URM_validation, URM_train, URM_test)


            # Done get the sparse matrices in the correct dictionary with the correct name
            # Done ICM_DICT and UCM_DICT can be empty if no ICMs or UCMs are required
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

            print("Movielens20MReader: loading complete")



