#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/11/18

@author: Maurizio Ferrari Dacrema
"""


from Recommenders.DataIO import DataIO

from time import time

import gdown as gd
from Conferences.IGCN_CF.igcn_cf_github.config import get_gowalla_config, get_yelp_config, get_amazon_config
from Conferences.IGCN_CF.igcn_cf_our_interface.DatasetPublic.IGN_CFReader import adjacencyList2COO,init_file_and_device,acquire_dataset,preprocessing

import os
import torch

class YelpReader(object):
    URM_DICT = {}
    ICM_DICT = {}

    def __init__(self, pre_splitted_path):

        super(YelpReader, self).__init__()
        device = torch.device('cuda')

        config = get_yelp_config(device)
        pre_splitted_path = pre_splitted_path+"Yelp/"

        dataIO = DataIO(pre_splitted_path)  ##initialize cool data manager

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):  ##avoid eventual crash if directory doesn't exist
            os.makedirs(pre_splitted_path)

        pre_splitted_filename = 'time.zip'

        print("YelpReader: Attempting to load pre-splitted data")
        try:
            print("YelpReader: Attempting to load pre-splitted data")

            ##attrib name is file name
            ##attrib object is panda object

            # all files should become like ./Gowalla/time.zip
            for attrib_name, attrib_object in dataIO.load_data(pre_splitted_filename).items():
                self.__setattr__(attrib_name, attrib_object)
                print(attrib_name, attrib_object)



        except FileNotFoundError:

            print("YelpReader: Pre-splitted data not found, building new one")

            print("YelpReader: loading URM")

            if not os.path.exists("Data_manager_split_datasets"):  ##avoid eventual crash if directory doesn't exist
                os.makedirs("Data_manager_split_datasets")

            url = "https://drive.google.com/u/0/uc?id=1QsGyi1UO844_jxd_r6PQpf7UHx8PayvL&export=download&confirm=no_antivirus"

            import requests
            req = requests.get(url)

            filename = "Data_manager_split_datasets/dataset.zip"

            # Writing the file to the local file system
            with open(filename, 'wb') as output_file:
                output_file.write(req.content)
            print('Downloading Completed')



            """"
            THIS STEP IS NEEDED TO CORRECTLY CREATE THE OBJECT TO CALL get_dataset IN dataset.py

            THIS CODE IS FROM run.py FROM ORIGINAL IMPLEMENTATION
            THIS IS A TWEAKED VERSION TO DECOUPLE THE CONFIG SPAWNING
            AND LET THE ORIGINAL METHODS FUNCTION PROPERLY
            """

            device, log_path = init_file_and_device()

            ###THIS CODE IS FROM run.py FROM ORIGINAL IMPLEMENTATION
            ##THIS IS A TWEAKED VERSION TO DECOUPLE THE CONFIG SPAWNING
            ##AND LET THE ORIGINAL METHODS FUNCTION PROPERLY

            """
            FIX RUNTIME CONFIG TO COMPLY WITH recsys_port README.md
            """
            """"
            CONFIG IS NEEDED TO USE THE get_dataset method from original dataset.py
            IT USES A sys.module REFERENCE SO IT IS NECESSARY TO CREATE A dataset.py LOCAL FILE
            WITH THE CORRECT import OR CHANGE THE ORIGINAL SOURCE CODE
            """

            # fix runtime config to comply with recsys_port README.md
            config[0][0]["path"] = "Data_manager_split_datasets/Yelp/time"

            # DO Replace this with the publicly available dataset you need
            # The DataManagers are in the Data_Manager folder, if the dataset is already there use that data reader

            import zipfile
            with zipfile.ZipFile("Data_manager_split_datasets/dataset.zip", 'r') as zip_ref:
                zip_ref.extractall("Data_manager_split_datasets/")

            dataset = acquire_dataset(log_path, config)

            from scipy import sparse
            n_items = 42706
            n_users = 75173



            datas, rows, cols = adjacencyList2COO(dataset.test_data)
            URM_test = sparse.coo_matrix((datas, (rows, cols)), shape=(n_users, n_items))

            datas, rows, cols = adjacencyList2COO(dataset.val_data)
            URM_val = sparse.coo_matrix((datas, (rows, cols)), shape=(n_users, n_items))

            datas, rows, cols = adjacencyList2COO(dataset.train_data)
            URM_train = sparse.coo_matrix((datas, (rows, cols)), shape=(n_users, n_items))

            print(URM_train.shape)
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

            print("YelpReader: loading complete")
