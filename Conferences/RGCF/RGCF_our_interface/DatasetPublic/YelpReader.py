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
import gdown as gd
import zipfile
from  Conferences.RGCF.RGCF_our_interface.DatasetPublic.RGCF_Reader import preprocessing_ratings
from Conferences.RGCF.RGCF_our_interface.DatasetPublic.RGCF_Reader import *
from recbole.utils import init_seed


class YelpReader(object):

    URM_DICT = {}
    ICM_DICT = {}

    def __init__(self, pre_splitted_path):

        super(YelpReader, self).__init__()

        config=get_config_preproc('yelp2018')
        config.dataset='yelp2018'
        config.final_config_dict['data_path']="Conferences/RGCF/RGCF_github/dataset/yelp2018"
        init_seed(config['seed'], config['reproducibility'])
        config.dataset='yelp2018'
        config.final_config_dict['data_path']="Conferences/RGCF/RGCF_github/dataset/yelp2018"
        init_seed(config['seed'], config['reproducibility'])
        config.final_config_dict["user_inter_num_interval"]= "[15,inf)"
        config.final_config_dict["item_inter_num_interval"]= "[15,inf)"

        #pre_splitted_path += "data_split/"
        pre_splitted_filename = "yelp2018-processed"

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        dataIO = DataIO(pre_splitted_path)

        try:
            print("YelpReader: Attempting to load pre-splitted data")

            for attrib_name, attrib_object in dataIO.load_data(pre_splitted_filename).items():
                 self.__setattr__(attrib_name, attrib_object)


        except FileNotFoundError:

            print("YelpReader: Pre-splitted data not found, building new one")


            if not os.path.exists("Data_manager_split_datasets"):  ##avoid eventual crash if directory doesn't exist
                os.makedirs("Data_manager_split_datasets")
            print("YelpReader: loading URM")




            ##Official recbole link may be busy




            filename = "Data_manager_split_datasets/Yelp_RGCF.zip"

            url="https://drive.google.com/u/0/uc?id=1Hte_6IDyqy-1Fjs6ArIqKp1NzGE4FcVn&export=download&confirm=no_antivirus"

            import requests
            req = requests.get(url)


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


            """
            This is the native RecBole function used to preprocess the data. Removing interactions
            and ratings under a certain threshold. The "True" parameters allows us to save
            these DataLoaders to reload them in the wrapper. Of course verifying the correctness.
            The ideal solution would have been passing in to the Wrapper explicitly, but the DataLoader object
            is not compatible with DataIO
            """
            train_data, valid_data, test_data = data_preparation(config, dataset, True)

            URM_train = train_data.dataset.inter_matrix(form='coo')
            URM_validation = valid_data.dataset.inter_matrix(form='coo')
            URM_test = test_data.dataset.inter_matrix(form='coo')


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
                # "train_data":train_data <---------This is not serializable by DataIO
                "ICM_DICT": self.ICM_DICT,
                "UCM_DICT": self.UCM_DICT,
                "URM_DICT": self.URM_DICT,
            }

            dataIO.save_data(pre_splitted_filename, data_dict_to_save=data_dict_to_save)

            print("YelpMReader: loading complete")



