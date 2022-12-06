#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/12/18

@author: Maurizio Ferrari Dacrema
"""
import os

from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.BaseTempFolder import BaseTempFolder
from Recommenders.DataIO import DataIO
from recbole.data import load_split_dataloaders
import numpy as np
import scipy.sparse as sps

from Conferences.RGCF.RGCF_github.trainer import customized_Trainer

from Conferences.RGCF.RGCF_github.rgcf import RGCF

from recbole.config import Config
import sys
import traceback
from recbole.data import create_dataset, data_preparation


class Params():
    lambda_u = 0
    lambda_v = 0
    lambda_r = 0
    a = 0
    b = 0
    M = 0
    n_epochs = 0


# Done replace the recommender class name with the correct one
class RGCF_RecommenderWrapper(BaseRecommender, Incremental_Training_Early_Stopping, BaseTempFolder):
    # Done replace the recommender name with the correct one
    RECOMMENDER_NAME = "RGCF_RecommenderWrapper"

    def __init__(self, URM_train):
        # Done remove ICM_train and inheritance from BaseItemCBFRecommender if content features are not needed
        # This is used in _compute_item_score
        super().__init__(URM_train)
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
            toPredict = {"user_id": user_id}

            scores = self.model.full_sort_predict(toPredict)

            item_score_user = scores.cpu().detach().numpy()
            # Do not modify this
            # Put the predictions in the correct items
            if items_to_compute is not None:
                item_scores[user_index, items_to_compute] = item_score_user.ravel()[items_to_compute]
            else:
                item_scores[user_index, :] = item_score_user.ravel()

        return item_scores

    """
        From URM to file Recbole 
    """
    def fromURMToRecbole(self, name, path):

        if(hasattr(self,"URM_val")):
            URM_val = self.URM_val.tocoo(copy=True)
            URM_test= self.URM_test.tocoo(copy=True)
            URM_train = self.URM_train.tocoo(copy=True)
            URM=URM_val+URM_test+URM_train
        else:
            URM_train = self.URM_train.tocoo(copy=True)
            URM=URM_train
        URM = URM.tocoo()

        path = os.path.join(path, name)
        file = open(path + ".inter", "w")
        fileu = open(path + ".user", "w")
        filei = open(path + ".item", "w")
        file.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")

        column = (URM.col).copy()
        row = (URM.row).copy()
        number_users = np.unique(row)
        number_items = np.unique(column)
        index=1
        missing = []
        for i in range(number_items[0],URM.shape[1]-1):
            if(i not in number_items):
                missing.append(i)


        print(len(np.unique(column)))
        for i in range(len(number_users)):
            count = np.count_nonzero(row == i)
            items_to_add = column[:count]
            items = items_to_add
            column = column[count:]
            for j in range(len(items)):
                # only users and items matters because the preprocessing is already done in the reader
                file.write(str(i+1) + "\t" + str(items[j]) + "\t" + "5.0" + "\t" + "5.0" + "\n")

        file.close()
        fileu.close()
        filei.close()


    def _init_model(self):
        """
        This function instantiates the model, it should only rely on attributes and not function parameters
        It should be used both in the fit function and in the load_model function
        :return:
        """
        config = Config(model=RGCF, dataset="DONT-CARE",
                        config_file_list=['./Conferences/RGCF/RGCF_github/config/data.yaml',
                                        './Conferences/RGCF/RGCF_github/config/model-rgcf.yaml'])
        config.final_config_dict['load_col'] = {'inter': ['user_id', 'item_id', 'rating'], 'item': ['item_id', 'genre']}
        config.internal_config_dict['load_col'] = {'inter': ['user_id', 'item_id', 'rating'],
                                                   'item': ['item_id', 'genre']}
        config.final_config_dict['val_interval'] = {'rating': '[3,inf)'}
        config.internal_config_dict['val_interval'] = {'rating': '[3,inf)'}
        config.final_config_dict['metrics'] = ['Recall', 'MRR', 'NDCG', 'Hit']
        config.internal_config_dict['metrics'] = ['Recall', 'MRR', 'NDCG', 'Hit']
        config.final_config_dict['training_neg_sample_num'] = 1
        config.internal_config_dict['training_neg_sample_num'] = 1
        config.final_config_dict['epochs'] = 500
        config.internal_config_dict['epochs'] = 500
        config.final_config_dict['train_batch_size'] = 4096
        config.internal_config_dict['train_batch_size'] = 4096
        config.final_config_dict['topk'] = [10, 20, 50]
        config.internal_config_dict['topk'] = [10, 20, 50]
        config.final_config_dict['save_dataloaders'] = True
        config.internal_config_dict['save_dataloaders'] = True

        train_data, test_data, val_data = load_split_dataloaders("./saved/ml-1m-for-RGCF-dataloader.pth")
        print(train_data.dataset)
        #dataset_name = "datasetRecbole"
        #dataset_path = "./Conferences/RGCF/RGCF_github/dataset"
        #self.URM_train.eliminate_zeros()
        #a=self.URM_train.tocoo(copy=True)
        #b=np.unique(a.col)
        #self.fromURMToRecbole(dataset_name, dataset_path)

        #config = Config(model=RGCF, dataset=dataset_name,
        #                config_file_list=['./Conferences/RGCF/RGCF_github/config/data.yaml',
        #                                  './Conferences/RGCF/RGCF_github/config/model-rgcf.yaml'])
        #config.final_config_dict['data_path'] = dataset_path
        #config.final_config_dict['normalize_field'] = False
        #config.internal_config_dict['normalize_field'] = False
        #config.final_config_dict['normalize_all'] = False
        #config.internal_config_dict['normalize_all'] = False
        # to pass to recbole only the trained data
        #config.internal_config_dict['eval_args']['split'] = {'RS': [1.0, 0, 0]}

        #dataset = create_dataset(config)

        #train_data, valid_data, test_data = data_preparation(config, dataset)
        print(train_data.dataset)
        model = RGCF

        train_data, test_data, val_data = load_split_dataloaders("./saved/"+self.dataset_name+"-for-RGCF-dataloader.pth")
        check=train_data.dataset.inter_matrix(form='csr')

        try:
            assert np.all(check.indices == self.URM_train.indices)
            assert np.all(check.indptr == self.URM_train.indptr)
        except AssertionError:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)  # Fixed format
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]

            print('DataLoader representing the URM_train is corrupted.')
            exit(1)

        self.model = model(config, train_data.dataset).to(config['device'])
        self.trainer = customized_Trainer(config, self.model)
        self.train_data = train_data
        print(self._compute_item_score([1,10,15]))
        print(self.model)
        print(self.trainer)

    def fit(self,
            article_hyperparameters=None,
            epochs=300,
            # Done replace those hyperparameters with the ones you need
            learning_rate_vae=1e-2,
            learning_rate_cvae=1e-3,
            num_factors=50,
            dimensions_vae=[200, 100],
            epochs_vae=[50, 50],
            batch_size=128,
            lambda_u=0.1,
            lambda_v=10,
            lambda_r=1,
            a=1,
            b=0.01,
            M=300,

            # These are standard
            temp_file_folder=None,
            **earlystopping_kwargs
            ):

        # Get unique temporary folder
        self.temp_file_folder = self._get_unique_temp_folder(input_temp_file_folder=temp_file_folder)

        # Done replace the following code with what needed to create an instance of the model.
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
        self._init_model()

        # Done Close all sessions used for training and open a new one for the "_best_model"
        # close session tensorflow

        ###############################################################################
        ### This is a standard training with early stopping part, most likely you won't need to change it

        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name=self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self.load_model(self.temp_file_folder, file_name="_best_model")

        self._clean_temp_folder(temp_file_folder=self.temp_file_folder)

        print("{}: Training complete".format(self.RECOMMENDER_NAME))

    def _prepare_model_for_validation(self):
        # Done Most likely you won't need to change this function
        pass

    def _update_best_model(self):
        # Done Most likely you won't need to change this function
        self.save_model(self.temp_file_folder, file_name="_best_model")

    def _run_epoch(self, currentEpoch):
        # Done replace this with the train loop for one epoch of the model

        total_loss = self.trainer.train_epoch(self.train_data, currentEpoch)
        print(" loss=%.5f" % (total_loss))

        return

    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        self.model.save(folder_path+"/_weights")
        data_dict_to_save = {
            # TODO replace this with the hyperparameters and attribute list you need to re-instantiate
            #  the model when calling the load_model
            "n_users": self.n_users,
            "n_items": self.n_items,
            "dataset_name":self.dataset_name,
            #"train_data":self.train_data,
            #"mf_dim": self.mf_dim,
            #"layers": self.layers,
            #"reg_layers": self.reg_layers,
            #"reg_mf": self.reg_mf,
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

        self._init_model()
        self.model.load(folder_path+"/_weights")


        self._print("Loading complete")
