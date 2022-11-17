#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/11/18

@author: Maurizio Ferrari Dacrema
"""
import time

from Conferences.IGN_CF.igcncf_github.dataset import ProcessedDataset
from Data_manager.DataReader import DataReader
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

from Recommenders.DataIO import DataIO
import networkx
import gdown as gd
from Conferences.IGN_CF.igcncf_github.config import get_gowalla_config, get_yelp_config, get_amazon_config
from Conferences.IGN_CF.igcncf_our_interface.DatasetPublic.IGN_CFReader import adjacencyList2COO,init_file_and_device,acquire_dataset,preprocessing

import os
import numpy as np
import scipy.sparse as sparse

class GowallaReader(DataReader):
    URM_DICT = {}
    ICM_DICT = {}

    def __init__(self, pre_splitted_path,config):

        super(GowallaReader, self).__init__()

        config = config

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


        except FileNotFoundError:

            print("GowallaReader: Pre-splitted data not found, building new one")

            print("GowallaReader: loading URM")

            url = "https://drive.google.com/file/d/1l7HJgrA2aYc8ZGExXUAx1Btr7QOOd-3b/view?usp=sharing"
            output = "Data_manager_split_datasets/dataset.zip"

            if not os.path.exists("Data_manager_split_datasets"):  ##avoid eventual crash if directory doesn't exist
                os.makedirs("Data_manager_split_datasets")

            if os.path.isfile(output) != True:
                gd.download(url=url, output=output, quiet=False, fuzzy=True)

            device, log_path = init_file_and_device()

            ###THIS CODE IS FROM run.py FROM ORIGINAL IMPLEMENTATION
            ##THIS IS A TWEAKED VERSION TO DECOUPLE THE CONFIG SPAWNING
            ##AND LET THE ORIGINAL METHODS FUNCTION PROPERLY

            # fix runtime config to comply with recsys_port README.md
            config[0][0]["path"] = 'Data_manager_split_datasets/Gowalla/time'

            #{'name': 'IGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device(type='cpu'), 'dropout': 0.3, 'feature_ratio': 1.0} gowala
            #{'name': 'IGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device(type='cpu'), 'dropout': 0.0, 'feature_ratio': 1.0} amz
            #{'name': 'IGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device(type='cpu'), 'dropout': 0.3, 'feature_ratio': 1.0}
            # DO Replace this with the publicly available dataset you need
            # The DataManagers are in the Data_Manager folder, if the dataset is already there use that data reader

            import zipfile
            with zipfile.ZipFile("Data_manager_split_datasets/dataset.zip", 'r') as zip_ref:
                zip_ref.extractall("Data_manager_split_datasets/")

            dataset = acquire_dataset(log_path, config)

            n_items = 40988
            n_users = 29858

            datas, rows, cols = adjacencyList2COO(dataset.val_data)
            URM_val = sparse.coo_matrix((datas, (rows, cols)), shape=(n_users, n_items))

            datas, rows, cols = adjacencyList2COO(dataset.train_data)
            URM_train = sparse.coo_matrix((datas, (rows, cols)), shape=(n_users, n_items))

            datas, rows, cols = adjacencyList2COO(dataset.test_data)
            URM_test = sparse.coo_matrix((datas, (rows, cols)), shape=(n_users, n_items))

            # Done Apply data preprocessing if required (for example binarizing the data, removing users ...)
            # we checked if the preprocessing is correct or not
            # binarize the data (only keep ratings >= 4)

            URM_val, URM_train, URM_test = preprocessing(n_users, n_items, URM_val, URM_train, URM_test)



            # Useless because we already have presplitted data select the data splitting that you need, almost certainly there already is a function that does the splitting
            #  in the way you need, if you are not sure, ask me via email
            # Split the data in train, validation and test
            # URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.8)
            # URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.8)

            # Done get the sparse matrices in the correct dictionary with the correct name
            # Done ICM_DICT and UCM_DICT can be empty if no ICMs or UCMs are required -> it's this case
            self.ICM_DICT = {}
            self.UCM_DICT = {}

            self.URM_DICT = {
                "URM_train": URM_train,
                "URM_test": URM_test,
                "URM_validation": URM_val,
            }

            # You likely will not need to modify this part
            data_dict_to_save = {
                "ICM_DICT": self.ICM_DICT,
                "UCM_DICT": self.UCM_DICT,
                "URM_DICT": self.URM_DICT,
            }

            dataIO.save_data(pre_splitted_filename, data_dict_to_save=data_dict_to_save)

            print("GowallaReader: loading complete")
