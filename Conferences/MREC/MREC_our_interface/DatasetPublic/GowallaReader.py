#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/11/18

@author: Maurizio Ferrari Dacrema
"""

import os
import gdown as gd
from Recommenders.DataIO import DataIO
import pandas as pd
from Conferences.MREC.MREC_our_interface.DatasetPublic.MRECReader import preprocessing_interactions_pandas

class GowallaReader():
    URM_DICT = {}
    ICM_DICT = {}

    def __init__(self, pre_splitted_path):

        super(GowallaReader, self).__init__()

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
        pre_splitted_filename = 'gowalla_MREC.zip'

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


            if not os.path.exists("Data_manager_split_datasets"):  ##avoid eventual crash if directory doesn't exist
                os.makedirs("Data_manager_split_datasets")

            filename = "Data_manager_split_datasets/Gowalla_totalCheckins.zip"

            url = "https://drive.google.com/u/0/uc?id=1-0Yt5TAC9QM4fCXv7FDY_f2rwC_0AW_F&export=download&confirm=no_antivirus"

            import requests
            req = requests.get(url)


            # Writing the file to the local file system
            with open(filename, 'wb') as output_file:
                output_file.write(req.content)
            print('Downloading Completed')

            import zipfile
            with zipfile.ZipFile("Data_manager_split_datasets/Gowalla_totalCheckins.zip", 'r') as zip_ref:
                zip_ref.extractall("Data_manager_split_datasets/Gowalla")

            """"
            THIS STEP IS NEEDED TO CORRECTLY CREATE THE OBJECT TO CALL get_dataset IN dataset.py

            THIS CODE IS FROM run.py FROM ORIGINAL IMPLEMENTATION
            THIS IS A TWEAKED VERSION TO DECOUPLE THE CONFIG SPAWNING
            AND LET THE ORIGINAL METHODS FUNCTION PROPERLY
            """




            dataset = pd.read_csv("Data_manager_split_datasets/Gowalla/Gowalla_totalCheckins.txt", sep='\t')

            dataset.columns = ['user_id', 'timestamp', 'long', 'lat','item_id']
            del dataset["timestamp"]
            del dataset["long"]
            del dataset["lat"]

            URM_train, URM_test= preprocessing_interactions_pandas(dataset,10)


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

            print("GowallaReader: loading complete")
