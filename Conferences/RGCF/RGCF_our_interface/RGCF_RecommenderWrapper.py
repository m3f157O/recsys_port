#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/12/18

@author: Maurizio Ferrari Dacrema
"""


from Recommenders.BaseCBFRecommender import BaseItemCBFRecommender
from Recommenders.BaseSimilarityMatrixRecommender import BaseUserSimilarityMatrixRecommender, \
    BaseSimilarityMatrixRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.BaseTempFolder import BaseTempFolder
from Recommenders.DataIO import DataIO

import numpy as np
import tensorflow as tf
import scipy.sparse as sps

from Conferences.CIKM.ExampleAlgorithm_github.main import get_model
from Conferences.RGCF.RGCF_github.trainer import customized_Trainer


from Conferences.RGCF.RGCF_github.rgcf import RGCF
class Params():
    lambda_u = 0
    lambda_v = 0
    lambda_r = 0
    a = 0
    b = 0
    M = 0
    n_epochs = 0

# Done replace the recommender class name with the correct one
class RGCF_RecommenderWrapper(BaseItemCBFRecommender, Incremental_Training_Early_Stopping, BaseTempFolder):

    # Done replace the recommender name with the correct one
    RECOMMENDER_NAME = "RGCF_RecommenderWrapper"

    def __init__(self, URM_train, ICM_train):
        # Done remove ICM_train and inheritance from BaseItemCBFRecommender if content features are not needed
        super(BaseUserSimilarityMatrixRecommender, self).__init__(URM_train)

        # This is used in _compute_item_score
        super().__init__(URM_train, ICM_train)
        self._item_indices = np.arange(0, self.n_items, dtype=np.int)


    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # Done if the model in the end is either a matrix factorization algorithm or an ItemKNN/UserKNN
        #  you can have this class inherit from BaseMatrixFactorization, BaseItemSimilarityMatrixRecommender
        #  or BaseUSerSimilarityMatrixRecommender
        #  in which case you do not have to re-implement this function, you only need to set the
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


        for user_index in range(len(user_id_array)):

            user_id = user_id_array[user_index]

            # TODO this predict function should be replaced by whatever code is needed to compute the prediction for a user

            # The prediction requires a list of two arrays user_id, item_id of equal length
            # To compute the recommendations for a single user, we must provide its index as many times as the
            # number of items
            item_score_user = self.model.predict(user_id)

            # Do not modify this
            # Put the predictions in the correct items
            if items_to_compute is not None:
                item_scores[user_index, items_to_compute] = item_score_user.ravel()[items_to_compute]
            else:
                item_scores[user_index, :] = item_score_user.ravel()


        return item_scores


    def _init_model(self,config,train_data):
        """
        This function instantiates the model, it should only rely on attributes and not function parameters
        It should be used both in the fit function and in the load_model function
        :return:
        """


        # TODO Instantiate the model
        # Always clear the default graph if using tehsorflow

        model = RGCF
        self.model = model(config, train_data.dataset).to(config['device'])
        self.trainer = customized_Trainer(config, model)

    def fit(self,
            epochs = 100,
            article_hyperparameters=None,
            # TODO replace those hyperparameters with the ones you need
            learning_rate_vae = 1e-2,
            learning_rate_cvae = 1e-3,
            num_factors = 50,
            dimensions_vae = [200, 100],
            epochs_vae = [50, 50],
            batch_size = 128,
            lambda_u = 0.1,
            lambda_v = 10,
            lambda_r = 1,
            a = 1,
            b = 0.01,
            M = 300,

            # These are standard
            temp_file_folder = None,
            **earlystopping_kwargs
            ):


        # Get unique temporary folder
        self.temp_file_folder = self._get_unique_temp_folder(input_temp_file_folder=temp_file_folder)


        # TODO replace the following code with what needed to create an instance of the model.
        #  Preferably create an init_model function
        #  If you are using tensorflow before creating the model call tf.reset_default_graph()

        # The following code contains various operations needed by another wrapper


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
            end_pos = self.URM_train.indptr[user_index +1]

            user_profile = self.URM_train.indices[start_pos:end_pos]
            self._train_users.append(list(user_profile))


        self._train_items = []

        self.URM_train = sps.csc_matrix(self.URM_train)

        for user_index in range(self.n_items):

            start_pos = self.URM_train.indptr[user_index]
            end_pos = self.URM_train.indptr[user_index +1]

            item_profile = self.URM_train.indices[start_pos:end_pos]
            self._train_items.append(list(item_profile))





        self.URM_train = sps.csr_matrix(self.URM_train)



        if(article_hyperparameters is not None):
            self._init_model(config=article_hyperparameters["config"],train_data=article_hyperparameters["train_data"])



        # TODO Close all sessions used for training and open a new one for the "_best_model"
        # close session tensorflow


        ###############################################################################
        ### This is a standard training with early stopping part, most likely you won't need to change it

        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)


        self.load_model(self.temp_file_folder, file_name="_best_model")

        self._clean_temp_folder(temp_file_folder=self.temp_file_folder)

        print("{}: Training complete".format(self.RECOMMENDER_NAME))



    def _prepare_model_for_validation(self):
        # TODO Most likely you won't need to change this function
        pass


    def _update_best_model(self):
        # TODO Most likely you won't need to change this function
        self.save_model(self.temp_file_folder, file_name="_best_model")





    def _run_epoch(self, currentEpoch):
        # TODO replace this with the train loop for one epoch of the model

        train_data=self.URM_train ##todo fix this
        show_progress=True
        self.model.train()

        loss_func = self.model.calculate_loss
        total_loss = None
        iter_data = (tqdm(
            enumerate(train_data),
            total=len(train_data),
            desc=set_color(f"Train {currentEpoch:>5}", 'pink'),
        ) if show_progress else enumerate(train_data))
        for batch_idx, interaction in iter_data:
            interaction = interaction.to(self.device)
            self.trainer.optimizer.zero_grad()

            loss = self.model.calculate_loss(interaction, epoch_idx=currentEpoch, tensorboard=self.tensorboard)

            total_loss = loss.item() if total_loss is None else total_loss + loss.item()
            self.trainer._check_nan(loss)
            loss.backward()

            self.trainer.optimizer.step()

        print("[#epoch=%06d], loss=%.5f" % (
            currentEpoch, total_loss))








    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        # TODO replace this with the Saver required by the model
        #  in this case the neural network will be saved with the _weights suffix, which is rather standard
        self.model.save_weights(folder_path + file_name + "_weights", overwrite=True)

        # TODO Alternativley you may save the tensorflow model with a session
        saver = tf.train.Saver()
        saver.save(self.sess, folder_path + file_name + "_session")

        data_dict_to_save = {
            # TODO replace this with the hyperparameters and attribute list you need to re-instantiate
            #  the model when calling the load_model
            "n_users": self.n_users,
            "n_items": self.n_items,
            "mf_dim": self.mf_dim,
            "layers": self.layers,
            "reg_layers": self.reg_layers,
            "reg_mf": self.reg_mf,
        }

        # Do not change this
        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self._print("Saving complete")




    def load_model(self, folder_path, file_name = None):

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

        # TODO If you are using tensorflow, you may instantiate a new session here
        # TODO reset the default graph to "clean" the tensorflow state
        tf.reset_default_graph()
        saver = tf.train.Saver()
        saver.restore(self.sess, folder_path + file_name + "_session")


        self._print("Loading complete")

