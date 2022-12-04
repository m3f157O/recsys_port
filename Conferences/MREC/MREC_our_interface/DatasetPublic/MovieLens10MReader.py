#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/11/18

@author: Maurizio Ferrari Dacrema
"""

import os
import gdown as gd
from Recommenders.DataIO import DataIO
from Conferences.MREC.MREC_our_interface.DatasetPublic.MRECReader import preprocessing_interactions_pandas
import pandas as pd
class MovieLens10MReader():
    URM_DICT = {}
    ICM_DICT = {}

    def __init__(self, pre_splitted_path):

        super(MovieLens10MReader, self).__init__()


        dataIO = DataIO(pre_splitted_path)  ##initialize cool data manager

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):  ##avoid eventual crash if directory doesn't exist
            os.makedirs(pre_splitted_path)

        #TODO fix
        pre_splitted_filename = 'time.zip'

        try:
            raise FileNotFoundError
            print("MovieLens10MReader: Attempting to load pre-splitted data")

            ##attrib name is file name
            ##attrib object is panda object

            # all files should become like ./Gowalla/time.zip
            for attrib_name, attrib_object in dataIO.load_data(pre_splitted_filename).items():
                self.__setattr__(attrib_name, attrib_object)


        except FileNotFoundError:

            print("MovieLens10MReader: Pre-splitted data not found, building new one")

            print("MovieLens10MReader: loading URM")

            ##todo fix download
            #url = "https://drive.google.com/file/d/1l7HJgrA2aYc8ZGExXUAx1Btr7QOOd-3b/view?usp=sharing"
            #output = "Data_manager_split_datasets/dataset.zip"

            #if not os.path.exists("Data_manager_split_datasets"):  ##avoid eventual crash if directory doesn't exist
            #    os.makedirs("Data_manager_split_datasets")

            #if os.path.isfile(output) != True:
            #    gd.download(url=url, output=output, quiet=False, fuzzy=True)


            # todo fix zip
            import zipfile
            #with zipfile.ZipFile("DatasetPublic/ml-10M.zip", 'r') as zip_ref:
            #    zip_ref.extractall("DatasetPublic/")

            dataset = pd.read_csv("DatasetPublic/ml-10M/ratings.dat", sep='::')

            dataset.columns = ['user_id', 'item_id', 'rating', 'timestamp']
            del dataset["timestamp"]

            URM_train, URM_test = preprocessing_interactions_pandas(dataset, 10, "Conferences/MREC/MREC_github/test/dataset/","ml-10m")

            print(URM_all)

            # TODO assign urms -> no urm val

            #URM_train = sparse.coo_matrix((datas, (rows, cols)), shape=(n_users, n_items))


            # Done Apply data preprocessing if required (for example binarizing the data, removing users ...)
            # we checked if the preprocessing is correct or not
            # binarize the data (only keep ratings >= 4)


            # Done get the sparse matrices in the correct dictionary with the correct name
            # Done ICM_DICT and UCM_DICT can be empty if no ICMs or UCMs are required -> it's this case
            self.ICM_DICT = {}
            self.UCM_DICT = {}

            self.URM_DICT = {
                "URM_train": URM_train,
                "URM_test": URM_test,
                "URM_validation": [],
            }

            # You likely will not need to modify this part
            data_dict_to_save = {
                "ICM_DICT": self.ICM_DICT,
                "UCM_DICT": self.UCM_DICT,
                "URM_DICT": self.URM_DICT,
            }

            dataIO.save_data(pre_splitted_filename, data_dict_to_save=data_dict_to_save)

            print("MovieLens10MReader: loading complete")
