#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from  Conferences.RGCF.RGCF_our_interface.DatasetPublic.RGCF_Reader import preprocessing_ratings

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

class AmazonReader(object):

    URM_DICT = {}
    ICM_DICT = {}

    def __init__(self, pre_splitted_path, config):

        super(AmazonReader, self).__init__()

        #pre_splitted_path += "data_split/"
        pre_splitted_filename = "amazon-processed"

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


            url = "https://drive.google.com/file/d/1x5I2wHvKf2C4KxtczGHLNvofHX_G5fS3/view?usp=share_link"
            output = "Data_manager_split_datasets/Amazon_Books_RGCF.zip"

            if os.path.isfile(output) != True:
                gd.download(url=url, output=output, quiet=False, fuzzy=True)

            with zipfile.ZipFile(output, 'r') as zip_ref:
                zip_ref.extractall(pre_splitted_path)


            preprocessing_ratings(file=config.final_config_dict['data_path'], rate=3.0,filename="/amz.inter")

            dataset = create_dataset(config)
            train_data, valid_data, test_data = data_preparation(config, dataset)

            URM_train = train_data.dataset.inter_matrix(form='coo')
            URM_validation = valid_data.dataset.inter_matrix(form='coo')
            URM_test = test_data.dataset.inter_matrix(form='coo')

            n_users = URM_train.shape[0] - 1
            n_items = URM_train.shape[1]

            URM_validation, URM_train, URM_test = preprocessing_interactions(n_users, n_items, URM_validation, URM_train, URM_test)

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

            print("AmazonMReader: loading complete")



