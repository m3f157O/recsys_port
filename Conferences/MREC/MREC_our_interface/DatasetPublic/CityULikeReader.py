#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/11/18

@author: Maurizio Ferrari Dacrema
"""

import os
import gdown as gd
from scipy.io import matlab
import pandas as pd
from Recommenders.DataIO import DataIO
from Data_manager.Movielens.Movielens10MReader import *
from Conferences.MREC.MREC_our_interface.DatasetPublic.MRECReader import preprocessing_interactions_lists
from Conferences.MREC.MREC_our_interface.DatasetPublic.MRECReader import preprocessing_interactions_pandas

import csv


class CityULikeReader():
    URM_DICT = {}
    ICM_DICT = {}

    def __init__(self, pre_splitted_path):

        super(CityULikeReader, self).__init__()

        """"
        CONFIG IS NEEDED TO USE THE get_dataset method from original dataset.py
        IT USES A sys.module REFERENCE SO IT IS NECESSARY TO CREATE A dataset.py LOCAL FILE
        WITH THE CORRECT import OR CHANGE THE ORIGINAL SOURCE CODE
        """

        dataIO = DataIO(pre_splitted_path)  ##initialize cool data manager

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):  ##avoid eventual crash if directory doesn't exist
            os.makedirs(pre_splitted_path)

        # TODO fix
        pre_splitted_filename = 'time.zip'

        try:
            print("CityULikeReader: Attempting to load pre-splitted data")
            raise FileNotFoundError
            ##attrib name is file name
            ##attrib object is panda object

            # all files should become like ./Gowalla/time.zip
            for attrib_name, attrib_object in dataIO.load_data(pre_splitted_filename).items():
                self.__setattr__(attrib_name, attrib_object)


        except FileNotFoundError:

            print("CityULikeReader: Pre-splitted data not found, building new one")

            print("CityULikeReader: loading URM")

            # TODO LINK FROM WHERE I TOOK THE DATASET: https://github.com/js05212/citeulike-a
            # url = "https://drive.google.com/file/d/1l7HJgrA2aYc8ZGExXUAx1Btr7QOOd-3b/view?usp=sharing"
            output = "DatasetPublic/cityulike-a-master.zip"

            # if not os.path.exists("Data_manager_split_datasets"):  ##avoid eventual crash if directory doesn't exist
            #    os.makedirs("Data_manager_split_datasets")

            # if os.path.isfile(output) != True:
            #    gd.download(url=url, output=output, quiet=False, fuzzy=True)
            """"
            THIS STEP IS NEEDED TO CORRECTLY CREATE THE OBJECT TO CALL get_dataset IN dataset.py

            THIS CODE IS FROM run.py FROM ORIGINAL IMPLEMENTATION
            THIS IS A TWEAKED VERSION TO DECOUPLE THE CONFIG SPAWNING
            AND LET THE ORIGINAL METHODS FUNCTION PROPERLY
            """

            import zipfile
            with zipfile.ZipFile("DatasetPublic/citeulike-a-master.zip", 'r') as zip_ref:
                zip_ref.extractall("DatasetPublic/CityULike")

            # dataset = pd.read_csv('DatasetPublic/CityULike/citeulike-a-master/users.dat', sep='|', header=0, skipinitialspace=True)
            with open('DatasetPublic/CityULike/citeulike-a-master/users.dat', 'r') as input_file:
                lines = input_file.readlines()
                newLines = []
                for line in lines:
                    newLine = line.strip(' ').split()
                    newLines.append(newLine)

            URM_all = preprocessing_interactions_lists(newLines)

            URMtoPandasCsv("DatasetPublic/CityULike/citeulike-a-master/users-items.csv", URM_all)
            dataset = pd.read_csv("DatasetPublic/CityULike/citeulike-a-master/users-items.csv", sep="\t")
            URM_train, URM_test = preprocessing_interactions_pandas(dataset,10,"Conferences/MREC/MREC_github/test/dataset/")

            eng = matlab.engine.start_matlab()
            matlab_script_directory = os.getcwd() + "/Conferences/MREC/MREC_github/test"
            eng.cd(matlab_script_directory)
            eng.split_dataset_original(nargout=0)
            # URM_train = sparse.coo_matrix((datas, (rows, cols)), shape=(n_users, n_items))

            # Done Apply data preprocessing if required (for example binarizing the data, removing users ...)
            # we checked if the preprocessing is correct or not
            # binarize the data (only keep ratings >= 4)

            dataset = pd.read_csv("DatasetPublic/CiteULike/users.dat", sep='\t')

            dataset.columns = ['user_id', 'timestamp', 'long', 'lat', 'item_id']
            del dataset["timestamp"]
            del dataset["long"]
            del dataset["lat"]

            URM_all = preprocessing_interactions_pandas(dataset, 10)

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

            print("CityULikeReader: loading complete")
