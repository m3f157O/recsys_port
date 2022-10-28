#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/11/18

@author: Maurizio Ferrari Dacrema
"""
import numpy as np
import gdown as gd
import scipy.sparse as sp

from Conferences.IGN_CF.igcncf_github.dataset import AmazonDataset
from Recommenders.DataIO import DataIO
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
import os


class DumbReader(object):
    URM_DICT = {}
    ICM_DICT = {}

    def __init__(self, dataset_name):

        super(DumbReader, self).__init__()

        pre_splitted_filename = "amazon_processed"



        try:
            print("DumbReader: Attempting to load pre-splitted data")
            amazon_data=AmazonDataset({'name': 'ProcessedDataset', 'path': 'data/Gowalla/1', 'device': device(type='cuda')}
)
        except FileNotFoundError:

            print("DumbReader: Pre-splitted data not found, building new one")

            print("DumbReader: loading URM")

            # Done Replace this with the publicly available dataset you need
            #  The DataManagers are in the Data_Manager folder, if the dataset is already there use that data reader

            url = "https://drive.google.com/file/d/1l7HJgrA2aYc8ZGExXUAx1Btr7QOOd-3b/view?usp=sharing"
            output = "Conferences/IGN_CF/igcncf_our_interface/DatasetPublic/data/idiot.zip"
            gd.download(url=url, output=output, quiet=False, fuzzy=True)

            # TODO MUST BE CHECKED Apply data preprocessing if required (for example binarizing the data, removing users ...)
            # binarize the data (only keep ratings >= 3) and users and items must have more than 10 interactions
            # URM_all.data = URM_all.data >= 3.0
            # URM_all.eliminate_zeros()
            # remove users that have less than 10 interactions
            # for user_id in range(URM_all.data.shape[0]):
            #    start_pos = URM_all.tocsc.indptr[user_id]
            #    end_pos = URM_all.indptr[user_id + 1]
            #    if sum(URM_all.indices[start_pos:end_pos]) < 10:
            #        URM_all.tocsr.indices[start_pos:end_pos] = 0
            ## remove items that have less than 10 interactions
            # for item_id in range(URM_all.data.shape[1]):
            #    start_pos = URM_all.tocsr.indptr[item_id]
            #    end_pos = URM_all.indptr[item_id + 1]
            #    if sum(URM_all.tocsc.indices[start_pos:end_pos]) < 10:
            #        URM_all.tocsc.indices[start_pos:end_pos] = 0
            # URM_all.eliminate_zeros()

            # Done select the data splitting that you need, almost certainly there already is a function that does the splitting
            #  in the way you need, if you are not sure, ask me via email
            # Split the data in train 70, validation 10 and test 20

            # TODO get the sparse matrices in the correct dictionary with the correct name
            # TODO ICM_DICT and UCM_DICT can be empty if no ICMs or UCMs are required

            # You likely will not need to modify this part

            print("DumbReader: loading complete")
