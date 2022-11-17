#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/12/18

@author: Maurizio Ferrari Dacrema
"""
import logging
from Conferences.IGN_CF.igcncf_github.config import *
from Conferences.IGN_CF.igcncf_github.dataset import *
from Recommenders.BaseCBFRecommender import BaseItemCBFRecommender
from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.BaseTempFolder import BaseTempFolder
from Recommenders.DataIO import DataIO

import torch
import numpy as np
import tensorflow as tf
import scipy.sparse as sps

from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from model import get_model
from Conferences.IGN_CF.igcncf_github.trainer import get_trainer

class DatasetOriginal(BasicDataset):
    train_array = []
    train_data = []
    device = torch.device('cpu')
    n_items = 0
    lenght = 0
    n_users = 0
    def __len__(self):
        return len(self.train_array)
    ## ^^^ AS IN model.AuxiliaryDataset (wtf)



def from_matrix_to_adjlist(matrix):
    list = []
    column = (matrix.col).copy()
    row = (matrix.row).copy()
    number_users = np.unique(row)
    for i in range(len(number_users)):
        count = np.count_nonzero(row == i)
        items_to_add = column[:count]
        items = items_to_add
        column = column[count:]
        list.append(items)
    return list


def restoreTrainArray(matrix):
    list = []
    column = (matrix.col).copy()
    row = (matrix.row).copy()
    number_users = np.unique(column)
    for i in range(len(column)):
        list.append([row[i], column[i]])
    return list

class Params():
    lambda_u = 0
    lambda_v = 0
    lambda_r = 0
    a = 0
    b = 0
    M = 0
    n_epochs = 0

# Done replace the recommender class name with the correct one
class IGN_CF_RecommenderWrapper(BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping,
                                BaseTempFolder):
    # Done replace the recommender name with the correct one
    RECOMMENDER_NAME = "IGN_CF_RecommenderWrapper"
    dataset = []
    model_config = {}
    trainer_config = {}
    trainer = {}

    sess = tf.compat.v1.Session()
    def __init__(self, URM_train):
        # Done remove ICM_train and inheritance from BaseItemCBFRecommender if content features are not needed
        # The model uses Matrix Factorization, so will inherit from it
        super(BaseMatrixFactorizationRecommender, self).__init__(URM_train)

        # This is used in _compute_item_score
        self._item_indices = np.arange(0, self.n_items, dtype=np.int)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        #  Done if the model in the end is either a matrix factorization algorithm (INMO USES MF)
        #  Done you can have this class inherit from BaseMatrixFactorization, BaseItemSimilarityMatrixRecommender
        #  or BaseUSerSimilarityMatrixRecommender
        #  Done in which case you do not have to re-implement this function, you only need to set the
        #  USER_factors, ITEM_factors (see PureSVD) or W_Sparse (see ItemKNN) data structures in the FIT function
        # In order to compute the prediction the model may need a Session. The session is an attribute of this Wrapper.
        # There are two possible scenarios for the creation of the session: at the beginning of the fit function (training phase)
        # or at the end of the fit function (before loading the best model, testing phase)

        # Do not modify this
        # Create the full data structure that will contain the item scores
        item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf

        if items_to_compute is not None:
            item_indices = items_to_compute
        else:
            item_indices = self._item_indices

        ##todo fix this, maybe take data in batches
        toTorch = np.array(user_id_array)
        t = torch.from_numpy(toTorch)



        rep = self.model.get_rep()
        users_r = rep[t, :]
        all_items_r = rep[self.n_users:, :]
        scores = torch.mm(users_r, all_items_r.t())

        item_score_user = scores
        for user_index in range(len(user_id_array)):

            # to pass from tensors to numpy
            item_score_user = scores[user_index].detach().numpy()

            # Done this predict function should be replaced by whatever code is needed to compute the prediction for a user

            # The prediction requires a list of two arrays user_id, item_id of equal length
            # To compute the recommendations for a single user, we must provide its index as many times as the
            # number of items
            # taken from IGN_CF in model.predict -> it requires only the use to calculate the scores
            # Do not modify this
            # Put the predictions in the correct items

            if items_to_compute is not None:
                item_scores[user_index, items_to_compute] = item_score_user.ravel()[items_to_compute]
            else:
                item_scores[user_index, :] = item_score_user.ravel()

        return item_scores

    def create_dataset(self, dataset_original):
        """
        This function instantiates the model, it should only rely on attributes and not function parameters
        It should be used both in the fit function and in the load_model function
        :return:
        """
        # Done steal CORRECT MODEL CONFIG (config[2][1])
        # Done call get model with hardcoded stuff ;)
        self.dataset = dataset_original

    def set_config(self,modelconfig,trainerconfig):
        self.trainer_config = trainerconfig
        self.model_config = modelconfig


    def create_trainer(self):
        self.trainer=get_trainer(self.trainer_config, self.dataset, self.model)
        print(self.trainer)

    """
    This function instantiates the model, it should only rely on attributes and not function parameters
    It should be used both in the fit function and in the load_model function
    :return:
     """
    def _init_model(self):

        ##todo fix because it is a bad practice
        config=get_gowalla_config(device=torch.device('cpu'))
        dataset_config, model_config, trainer_config = config[2]
        orignalDataset = DatasetOriginal(dataset_config)

        self.trainer_config=trainer_config
        URM_coo=self.URM_train.tocoo(copy=True)
        orignalDataset.n_users = URM_coo.shape[0]
        orignalDataset.n_items = URM_coo.shape[1]
        orignalDataset.device = torch.device('cpu')
        orignalDataset.train_array = restoreTrainArray(URM_coo)
        orignalDataset.lenght = len(orignalDataset.train_array)
        orignalDataset.train_data = from_matrix_to_adjlist(URM_coo)

        self.dataset=orignalDataset

        self.model = get_model(model_config, orignalDataset)
        print(self.model)


    def fit(self,
            #default params
            learning_rate_vae=1e-2,
            learning_rate_cvae=1e-3,
            num_factors=50,
            dimensions_vae=[200, 100],
            epochs_vae=[50, 50],
            lambda_u=0.1,
            lambda_v=10,
            lambda_r=1,
            M=300,

            #new hyperparameters from the paper
            learning_rate=[0.0001, 0.001, 0.01],
            regularization_coefficient=[0, 0.00001, 0.0001, 0.001, 0.01],
            dropout_rate=[0, 0.1, 0.3, 0.5, 0.7, 0.9],
            sampling_size=50,
            batch_size=2048,
            a=1,
            b=0.01,
            epochs=1000,
            embedding_size=64,




            # These are standard
            temp_file_folder=None,
            **earlystopping_kwargs
            ):

        # Get unique temporary folder
        self.temp_file_folder = self._get_unique_temp_folder(input_temp_file_folder=temp_file_folder)
        self.USER_factors = np.array([num_factors * self.n_users])
        self.ITEM_factors = np.array([num_factors * self.n_items])

        # DONE replace the following code with what needed to create an instance of the model.
        #  Preferably create an init_model function
        #  If you are using tensorflow before creating the model call tf.reset_default_graph()
        self._init_model()

        self.create_trainer()
        # The following code contains various operations needed by another wrapper
        ##todo delete after debugging
        #self._compute_item_score(np.array([3,12]))


        self._params = Params()
        self._params.lambda_u = lambda_u
        self._params.lambda_v = lambda_v
        self._params.lambda_r = lambda_r
        self._params.a = a
        self._params.b = b
        self._params.M = M
        self._params.n_epochs = epochs
        # These are the train instances as a list of lists
        # The following code processed the URM into the data structure the model needs to train
        self._train_users = []

        self.URM_train = sps.csr_matrix(self.URM_train)

        for user_index in range(self.n_users):
            start_pos = self.URM_train.indptr[user_index]
            end_pos = self.URM_train.indptr[user_index + 1]

            user_profile = self.URM_train.indices[start_pos:end_pos]
            self._train_users.append(list(user_profile))

        self._train_items = []

        self.URM_train = sps.csc_matrix(self.URM_train)

        for user_index in range(self.n_items):
            start_pos = self.URM_train.indptr[user_index]
            end_pos = self.URM_train.indptr[user_index + 1]

            item_profile = self.URM_train.indices[start_pos:end_pos]
            self._train_items.append(list(item_profile))

        self.URM_train = sps.csr_matrix(self.URM_train)

        # Done Close all sessions used for training and open a new one for the "_best_model"
        # close session tensorflow
        self.sess.close()
        self.sess = tf.compat.v1.Session()

        ###############################################################################
        ### This is a standard training with early stopping part, most likely you won't need to change it

        self._update_best_model()
        print("[!][!][!] STARTING TRAINING WITH EARLY STOPPING [!][!][!]")
        self._train_with_early_stopping(epochs,
                                        algorithm_name=self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self.load_model(self.temp_file_folder, file_name="_best_model")

        self._clean_temp_folder(temp_file_folder=self.temp_file_folder)

        print("{}: Training complete".format(self.RECOMMENDER_NAME))

    def _prepare_model_for_validation(self):
        pass

    def _update_best_model(self):
        self.save_model(self.temp_file_folder, file_name="_best_model")

    def _run_epoch(self, currentEpoch):
        # Done??? replace this with the train loop for one epoch of the model

        avg_loss=self.trainer.train_one_epoch()

        logging.info("[#epoch=%06d], loss=%.5f, neg_likelihood=%.5f" % (
            currentEpoch, avg_loss, self.trainer.epoch,)) ##NO GEN LOSS SORRY gen_loss))

    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        # Done replace this with the Saver required by the model
        #  in this case the neural network will be saved with the _weights suffix, which is rather standard
        self.model.save(file_name)


        data_dict_to_save = {
            # TODO replace this with the hyperparameters and attribute list you need to re-instantiate
            #  the model when calling the load_model
            "n_users": self.n_users,
            "n_items": self.n_items,
        }

        # Do not change this
        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

        self._print("Saving complete")

    def load_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(folder_path + file_name))

        # Reload the attributes dictionary
        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)

        for attrib_name in data_dict.keys():
            self.__setattr__(attrib_name, data_dict[attrib_name])

        # TODO replace this with what required to re-instantiate the model and load its weights,
        #  Call the init_model function you created before
        self._init_model()
        self.model.load_weights(folder_path + file_name + "_weights")


        # Done If you are using tensorflow, you may instantiate a new session here
        # Done reset the default graph to "clean" the tensorflow state
        # the version v1 of compact must be used because other versions are deprecated
        tf.compat.v1.reset_default_graph()
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, folder_path + file_name + "_session")

        self._print("Loading complete")
