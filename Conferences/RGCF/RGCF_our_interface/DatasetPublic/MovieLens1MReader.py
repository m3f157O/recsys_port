#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/11/18

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.DataIO import DataIO
import os
from recbole.data import create_dataset, data_preparation
from Conferences.RGCF.RGCF_our_interface.DatasetPublic.RGCF_Reader import preprocessing_interactions
import zipfile
import gdown as gd
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


            filename = "Data_manager_split_datasets/ml-1m_RGCF.zip"
            ##this is a trick because google drive is bugged for files too small
            url="https://codeload.github.com/m3f157O/ml-1m/zip/refs/heads/main"

            import requests
            req = requests.get(url)

            if not os.path.exists("Data_manager_split_datasets"):
                os.makedirs("Data_manager_split_datasets")

            # Writing the file to the local file system
            with open(filename, 'wb') as output_file:
                output_file.write(req.content)
            print('Downloading Completed')


            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(pre_splitted_path)





            """
            This is the native RecBole function used to create the recbole.data.dataset.Dataset object.
            Unluckily, when parsing an already split URM, it cannot avoid reindexing of cold items.
            explained better below
            """
            dataset = create_dataset(config)
            print(dataset)
            # in MovieLens, is not required a pre-processing


            """
            This is the native RecBole function used to preprocess the data. Removing interactions
            and ratings under a certain threshold. The "True" parameters allows us to save
            these DataLoaders to reload them in the wrapper. Of course verifying the correctness.
            The ideal solution would have been passing in to the Wrapper explicitly, but the DataLoader object
            is not compatible with DataIO
            """
            train_data, valid_data, test_data = data_preparation(config, dataset,True)



            URM_train = train_data.dataset.inter_matrix(form='coo')
            URM_validation = valid_data.dataset.inter_matrix(form='coo')
            URM_test = test_data.dataset.inter_matrix(form='coo')


            #preprocessing DONE BY RECBOLE >.<
            #URM_validation, URM_train, URM_test = preprocessing_interactions(n_users, n_items, URM_validation, URM_train, URM_test)


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
                #"train_data":train_data <---------This is not serializable by DataIO
                "ICM_DICT": self.ICM_DICT,
                "UCM_DICT": self.UCM_DICT,
                "URM_DICT": self.URM_DICT,
            }

            dataIO.save_data(pre_splitted_filename, data_dict_to_save=data_dict_to_save)

            print("Movielens20MReader: loading complete")



