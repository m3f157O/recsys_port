#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/11/18

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.DataIO import DataIO
import os
from recbole.data import create_dataset, data_preparation
from Conferences.RGCF.RGCF_our_interface.DatasetPublic.RGCF_Reader import preprocessing
import gdown as gd
import zipfile
import shutil

class YelpReader(object):

    URM_DICT = {}
    ICM_DICT = {}

    def __init__(self, pre_splitted_path, config):

        super(YelpReader, self).__init__()

        #pre_splitted_path += "data_split/"
        pre_splitted_filename = "yelp-processed"

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        dataIO = DataIO(pre_splitted_path)

        try:

            raise FileNotFoundError
            print("YelpReader: Attempting to load pre-splitted data")

            for attrib_name, attrib_object in dataIO.load_data(pre_splitted_filename).items():
                 self.__setattr__(attrib_name, attrib_object)


        except FileNotFoundError:

            print("YelpReader: Pre-splitted data not found, building new one")

            ##todo find a way to download from logged-in kaggle account
            ##or upload on drive :)
            print("YelpReader: loading URM")

            url = "https://drive.google.com/file/d/1Hte_6IDyqy-1Fjs6ArIqKp1NzGE4FcVn/view"
            output = "Data_manager_split_datasets/Yelp_RGCF.zip"

            if os.path.isfile(output) != True:
                gd.download(url=url, output=output, quiet=False, fuzzy=True)

            with zipfile.ZipFile(output, 'r') as zip_ref:
                zip_ref.extractall(pre_splitted_path)
            #fix this broken dataset
            #if os.path.isfile(pre_splitted_path+'yelp2018') != True:
            #    shutil.move("./yelp2018",pre_splitted_path)

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

            print("YelpMReader: loading complete")



