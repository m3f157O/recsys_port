#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/11/18

@author: Maurizio Ferrari Dacrema
"""
from Data_manager.DataReader import DataReader
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

from Recommenders.DataIO import DataIO

from IGN_CFReader import *

from Conferences.IGN_CF.igcncf_github.config import get_gowalla_config, get_yelp_config, get_amazon_config

import numpy as np
import scipy.sparse as sp

import os


class AmazonReader(DataReader):
    URM_DICT = {}
    ICM_DICT = {}

    def __init__(self, pre_splitted_path):

        super(AmazonReader, self).__init__()

        pre_splitted_path = "DatasetPublic/data/Amazon/"  ##local path, as described in recsys_port README.md

        dataIO = DataIO(pre_splitted_path)  ##initialize cool data manager

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):  ##avoid eventual crash if directory doesn't exist
            os.makedirs(pre_splitted_path)

        pre_splitted_filename = 'time.zip'

        try:

            raise FileNotFoundError
            print("AmazonReader: Attempting to load pre-splitted data")

            for attrib_name, attrib_object in dataIO.load_data(pre_splitted_filename).items():
                self.__setattr__(attrib_name, attrib_object)


        except FileNotFoundError:

            super(GowallaReader, self).__init__()

            pre_splitted_path = "DatasetPublic/data/Gowalla/"  ##local path, as described in recsys_port README.md

            dataIO = DataIO(pre_splitted_path)  ##initialize cool data manager

            # If directory does not exist, create
            if not os.path.exists(pre_splitted_path):  ##avoid eventual crash if directory doesn't exist
                os.makedirs(pre_splitted_path)

            pre_splitted_filename = 'time.zip'

            ##txt_to_csv("DatasetPublic/data/Amazon/time")
            try:

                raise FileNotFoundError
                print("GowallaReader: Attempting to load pre-splitted data")

                ##attrib name is file name
                ##attrib object is panda object

                # all files should become like ./Gowalla/time.zip
                for attrib_name, attrib_object in dataIO.load_data(pre_splitted_filename).items():
                    self.__setattr__(attrib_name, attrib_object)
                    print(attrib_name, attrib_object)


            except FileNotFoundError:

                print("AmazonReader: Pre-splitted data not found, building new one")

                print("AmazonReader: loading URM")

                url = "https://drive.google.com/file/d/1l7HJgrA2aYc8ZGExXUAx1Btr7QOOd-3b/view?usp=sharing"
                output = "../../../Data_manager_split_datasets/dataset.zip"

                if os.path.isfile(output) != True:
                    gd.download(url=url, output=output, quiet=False, fuzzy=True)

                device, log_path = init_file_and_device()

                ###THIS CODE IS FROM run.py FROM ORIGINAL IMPLEMENTATION
                ##THIS IS A TWEAKED VERSION TO DECOUPLE THE CONFIG SPAWNING
                ##AND LET THE ORIGINAL METHODS FUNCTION PROPERLY
                config = get_gowalla_config(device)

                # fix runtime config to comply with recsys_port README.md
                config[0][0]["path"] = '../../../Data_manager_split_datasets/Amazon/time'

                # DO Replace this with the publicly available dataset you need
                # The DataManagers are in the Data_Manager folder, if the dataset is already there use that data reader

                import zipfile
                with zipfile.ZipFile("../../../Data_manager_split_datasets/dataset.zip", 'r') as zip_ref:
                    zip_ref.extractall("../../../Data_manager_split_datasets/")

                dataset = acquire_dataset(log_path, config)

                from scipy import sparse
                n_items = 40988
                n_users = 29858

                datas, rows, cols = adjacencyList2COO(dataset.val_data)

                URM_val = sparse.coo_matrix((datas, (rows, cols)), shape=(n_users, n_items))

                # Done Apply data preprocessing if required (for example binarizing the data, removing users ...)
                # we checked if the preprocessing is correct or not
                # binarize the data (only keep ratings >= 4)

                URM_val = URM_val >= 4.0

                start = time.time()
                URM_train = dataset.train_data
                URM_test = dataset.test_data
                URM_val = sparse.csr_matrix(URM_val)
                # URM_val.eliminate_zeros()
                print(str(URM_val[0]))
                for user_id in range(np.shape(URM_val)[0]):
                    # interactions = URM_val[user_id]
                    # start_pos = np.count_nonzero(interactions)
                    print("STARTPOS" + str(start_pos))
                    # summation = sum(URM_val.tocsr().indices[start_pos:end_pos])
                    # sum += sum(URM_val.tocsr().indices[start_pos:end_pos])

                    # if sum(URM_val.tocsr().indices[start_pos:end_pos]) < 10:
                    # print(summation)
                    # URM_val.tocsr().indices[start_pos:end_pos] = 0
                URM_val.eliminate_zeros()
                # for item_id in range(np.shape(URM_val)[0]):
                #    start_pos = URM_val.tocsc().indptr[item_id]
                #    end_pos = URM_val.tocsc().indptr[item_id + 1]
                #    if sum(URM_val.tocsc().indices[start_pos:end_pos]) < 10:
                #        URM_val.tocsc().indices[start_pos:end_pos] = 0
                # URM_val.eliminate_zeros()

                end = time.time()
                print(end - start)

                # Useless because we already have presplitted data select the data splitting that you need, almost certainly there already is a function that does the splitting
                #  in the way you need, if you are not sure, ask me via email
                # Split the data in train, validation and test
                # URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.8)
                # URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.8)

                # TODO get the sparse matrices in the correct dictionary with the correct name
                # TODO ICM_DICT and UCM_DICT can be empty if no ICMs or UCMs are required
                self.ICM_DICT = {}
                self.UCM_DICT = {}

                self.URM_DICT = {
                    "URM_train": dataset.train_data,
                    "URM_test": dataset.test_data,
                    "URM_validation": dataset.val_data,
                }

                # You likely will not need to modify this part
                data_dict_to_save = {
                    "ICM_DICT": self.ICM_DICT,
                    "UCM_DICT": self.UCM_DICT,
                    "URM_DICT": self.URM_DICT,
                }

                dataIO.save_data(pre_splitted_filename, data_dict_to_save=data_dict_to_save)

                print("AmazonReader: loading complete")
