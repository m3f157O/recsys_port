#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/12/18

@author: Maurizio Ferrari Dacrema
"""
import logging
from Conferences.IGCN_CF.igcn_cf_github.config import *
from Conferences.IGCN_CF.igcn_cf_github.dataset import *
from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.BaseTempFolder import BaseTempFolder
from Recommenders.DataIO import DataIO

import torch
import numpy as np
import scipy.sparse as sps

from model import get_model
from Conferences.IGCN_CF.igcn_cf_github.trainer import get_trainer


"""
    Builds the dataset original to convert from a matrix to the adjacent list to train the model 
"""
class DatasetOriginal(BasicDataset):
    train_array = []
    train_data = []
    device = torch.device('cuda')
    n_items = 0
    length = 0
    n_users = 0

    def __len__(self):
        return len(self.train_array)
    ## ^^^ AS IN model.AuxiliaryDataset (wtf)


"""
    Convert from a matrix to the adjacent list 
"""
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

"""
     Convert from a matrix to the customized list to train the model with the following format:
     tuple : (row[index], col[index])
"""
def restoreTrainArray(matrix):
    list = []
    column = (matrix.col).copy()
    row = (matrix.row).copy()
    number_users = np.unique(column)
    for i in range(len(column)):
        list.append([row[i], column[i]])
    return list

"""
    Class to define the parameters used in the fit model
"""
class Params():
    lambda_u = 0
    lambda_v = 0
    lambda_r = 0
    a = 0
    b = 0
    M = 0
    n_epochs = 0


# Done replace the recommender class name with the correct one
"""
    Wrapper of the algorithm IGCN 
"""
class IGCN_CF_RecommenderWrapper(BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping,
                                 BaseTempFolder):
    # Done replace the recommender name with the correct one
    RECOMMENDER_NAME = "IGN_CF_RecommenderWrapper"
    dataset = []
    model_config = {}
    trainer_config = {}
    trainer = {}

    #sess = tf.compat.v1.Session()

    """
        IGCN model uses only Matrix Factorization without ICM_train, so it inherits from 
        BaseMatrixFactorizationRecommender   
    """
    def __init__(self, URM_train):
        # Done remove ICM_train and inheritance from BaseItemCBFRecommender if content features are not needed
        # The model uses Matrix Factorization, so will inherit from it
        super(BaseMatrixFactorizationRecommender, self).__init__(URM_train)

        # This is used in _compute_item_score
        self._item_indices = np.arange(0, self.n_items, dtype=np.int)

    """
        Computes item scores for each user 
    """
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
            items_to_compute=np.sort(items_to_compute)
        else:
            item_indices = self._item_indices

        ##READ COMMENT BELOW
        self.model.training = False
        for user_index in range(len(user_id_array)):

            toTorch = np.array([user_id_array[user_index]])
            t = torch.from_numpy(toTorch)

            """
            FOR SOME REASON model.predicts HAS AN UNDEFINED BEHAVIOUR, THIS IS DUE
            TO THE FACT THAT MANY FLAGS ARE CHECKED IN THE ORIGINAL IMPLEMENTATION,
            MODIFYING THE WEIGHTS AND APPLYING DROPOUT AT PREDICTION-TIME
            
            WE JUST DECIDED TO CUT OUT THE EXACT SAME CODE USER FOR THE PREDICTION. MOREOVER, 
            IT IS NECESSARY TO SET THE TRAINING FLAG TO FALSE, OR, EVEN DURING CONSEQUENT PREDICTIONS,
            THE WEIGHT MAY CHANGE DRAMATICALLY
            """
            rep = self.model.get_rep()
            users_r = rep[t, :]
            all_items_r = rep[self.n_users:, :]
            scores = torch.mm(users_r, all_items_r.t())
            # to pass from tensors to numpy
            item_score_user = scores.cpu().detach().numpy()
            # taken from IGN_CF in model.predict -> it requires only the use to calculate the scores

            # Do not modify this
            # Put the predictions in the correct items

            if items_to_compute is not None:
                item_scores[user_index, items_to_compute] = item_score_user.ravel()[items_to_compute]
            else:
                item_scores[user_index, :] = item_score_user.ravel()


        return item_scores

    """
            This function instantiates the model, it should only rely on attributes and not function parameters
            It should be used both in the fit function and in the load_model function
            :return:
    """


    """
        This function instantiates the model, it should only rely on attributes and not function parameters
        It should be used both in the fit function and in the load_model function
        :return:
     """

    def _init_model(self):

        """

        WHAT DONE BELOW MAY SEEM LIKE A BAD PRACTICE, BUT WE ONLY DID IT TO INCREASE THE SPEED
        AND NOT TO WASTE TONS OF SPACE:

        WE WOULD NEED TO SAVE TWO CONFIG DICTIONARIES TO REINSTANTIATE THE MODEL AFTER SAVING IT:
        IT IS REALLY SLOW, BECAUSE THESE DICTIONARIES CONTAIN WHOLE DATA STRUCTURES WHICH ARE
        IN FACT NOT NEEDED TO REINSTANTIATE THE MODEL.

        INSTEAD WE DECIDED, GIVEN THAT THE TRAINER AND MODEL CONFIG
        ARE THE SAME FOR ALL THREE DATASET (EXCEPT FOR AMAZON WHICH HAS DROPOUT ZERO)
        TO  JUST USE THE GOWALLA CONFIGURATION, WHICH IS THE EXACT
        SAME AS THE OTHER TWO (THE DIFFERENCE IS THE DATASET BUT WE ALREADY HAVE THAT AS self.URM)
        EXCEPT FOR A DROPOUT VALUE IN model_config FOR Amazon (NO EXPLANATION OF THIS DIFFERENCE HAS BEEN
        GIVEN BY THE AUTHORS, PROBABLY DROPOUT WORSENS THE PERFORMANCES ON BIGGER DATASETS)

        SO, TO SPEED UP THE RELOADING AND AVOID SAVING HUGE DICTIONARIES, WE DECIDE TO USE THIS APPROACH.

        THE RESULT WILL BE:
        INSTANTIATE THE CORRECT MODEL AND TRAINER, WITH DIFFERENT CONFIG BUT RIGHT DATASET.
        (OPTIONAL) PUT DROPOUT TO ZERO IF IT'S AMAZON DATASET (EASILY DETECTED)
        USE THE BUILT IN MODEL FUNCTION TO RELOAD THE CORRECT WEIGHTS

        IF ITS NOT STRUCTURALLY CORRECT, IT WOULD ALSO BE EASY FOR US TO CORRECT THIS AND JUST SAVE
        ALL IN THE DataIO DICTIONARY
        JUST DECOMMENT ALL THE LINES IN ALL FILES BELOW THE FOLLOWING:
                ##DECOMMENT IF YOU WANT TO SAVE CONFIGS TO DataIO

        """


        config = get_gowalla_config(device=torch.device('cuda'))
        dataset_config, model_config, trainer_config = config[2]

        ##DECOMMENT IF YOU WANT TO SAVE CONFIGS TO DataIO
        #model_config=self.model_config
        #trainer_config=self.trainer_config
        datasetOriginalFormat = DatasetOriginal(dataset_config)
        URM_coo = self.URM_train.tocoo(copy=True)
        datasetOriginalFormat.n_users = URM_coo.shape[0]
        datasetOriginalFormat.n_items = URM_coo.shape[1]
        datasetOriginalFormat.device = torch.device('cuda')
        datasetOriginalFormat.train_array = restoreTrainArray(URM_coo)
        datasetOriginalFormat.length = len(datasetOriginalFormat.train_array)
        datasetOriginalFormat.train_data = from_matrix_to_adjlist(URM_coo)

        ###model needs dataset as attribute to train and predict

        if(datasetOriginalFormat.n_users>100000):
            model_config['dropout']=0.0
        self.model = get_model(model_config, datasetOriginalFormat)

        print(self.model)

        """
            Creates the trainer that has to be taken each epoch because using only the training loop would have implied
            to change a lot of code, so we were obliged to take each epoch the trainer
        """

        self.trainer = get_trainer(trainer_config, datasetOriginalFormat, self.model)
        print(self.trainer)


    """
        Function to instantiate and train the model 
    """

    def fit(self,
            article_hyperparameters=None,
            # default params
            learning_rate_vae=1e-2,
            learning_rate_cvae=1e-3,
            num_factors=50,
            dimensions_vae=[200, 100],
            epochs_vae=[50, 50],
            lambda_u=0.1,
            lambda_v=10,
            lambda_r=1,
            M=300,

            # new hyperparameters from the paper
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
            **earlystopping_kwargs,

            ):

        # Get unique temporary folder
        self.temp_file_folder = self._get_unique_temp_folder(input_temp_file_folder=temp_file_folder)


        # DONE replace the following code with what needed to create an instance of the model.
        #  Preferably create an init_model function
        #  If you are using tensorflow before creating the model call tf.reset_default_graph()

        """
        THIS MAY BE AN OPINABLE CHOICE, BUT, AS SAID BEFORE, THE FUNCTION WILL DEFAULT TO THE GOWALLA MODEL IF NO PARAMETERS ARE GIVEN.
        OF COURSE IF A WRAPPER WITH YELP DATASET CALLS INIT MODEL WITH DEFAULT PARAMETERS, IT IS ABSOLUTELY WRONG, BUT THIS NEVER HAPPENS.
        
        WHEN A MODEL IS SUBSEQUENTLY RELOADED, IT WILL ALWAYS LOAD THE CORRECT TRAINER (IGCN TRAINER), AND ALWAYS LOAD THE CORRECT MODEL,
        EXCEPT FOR AMAZON CONFIGURATION, BECAUSE IT DIFFERS FROM ALL THE OTHER MODELS (NO DROPOUT, NO EXPLANATION GIVEN BY AUTHORS)
        """

        ##DECOMMENT IF YOU WANT TO SAVE CONFIGS TO DataIO
        #self.model_config=article_hyperparameters['model_config']
        #self.trainer_config=article_hyperparameters['trainer_config']
        self._init_model()



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

        # Done Close all sessions used for training and open a new one for the "_best_model"
        # close session tensorflow
        #self.sess.close()
        #self.sess = tf.compat.v1.Session()

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

    """
        Saves the best model after the training phase
    """
    def _update_best_model(self):
        self.save_model(self.temp_file_folder, file_name="_best_model")

    """
        Train only one epoch
    """
    def _run_epoch(self, currentEpoch):
        # Done replace this with the train loop for one epoch of the model -> we couldn't because we would have to change
        # most of the code, so each epoch we retrieve the trainer and train the model



        """
        THIS IS NOT PROPERLY COMPLIANT TO THE REQUIREMENTS, BUT DECOUPLING THE TRAINER FROM THE MODEL
        WOULD HAVE MEANT CHANGING A LOT OF CODE AND MOREOVER NO GUARANTEES ON THE CORRECTNESS

        IN THE ORIGINAL IMPLEMENTATION, THE ADDITIONAL DATA STRUCTURES OTHER THAN THE MODEL ARE:
        trainer.dataloader, trainer.aux_dataloader, trainer.l2_reg, trainer.aux_reg,trainer.opt.zero_grad(),trainer.opt.step()
        WHICH WOULD BE REALLY HEAVY AND RISKY TO IMPLEMENT, AS THEY ARE ALSO ATTRIBUTES OF THE ABSTRACT CLASS BasicTrainer

        EXTREMELY HARD TO DECOUPLE
        """
        start = time.time()
        self.model.training = True

        avg_loss = self.trainer.train_one_epoch()
        end = time.time()

        print("loss=%.5f, duration:%ds" % (avg_loss,end-start))  ##NO GEN LOSS SORRY gen_loss))

        logging.info("loss=%.5f" % (avg_loss))  ##NO GEN LOSS SORRY gen_loss))

    """
        Saves the model after training it
    """
    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        # Done replace this with the Saver required by the model
        #  in this case the neural network will be saved with the _weights suffix, which is rather standard
        self.model.save(folder_path + file_name + "_weights")


        data_dict_to_save = {
            # Done replace this with the hyperparameters and attribute list you need to re-instantiate
            #  the model when calling the load_model
            "n_users": self.n_users,
            "n_items": self.n_items,
            # modified
            "batch_size": 2048,
            "epochs": 1000,
            "embedding_size": 64,

            # default
            "epochs_MFBPR": 500,
            "hidden_size": 128,
            "negative_sample_per_positive": 1,
            "negative_instances_per_positive": 4,
            "regularization_users_items": 0.01,
            "regularization_weights": 10,
            "regularization_filter_weights": 1,
            "learning_rate_embeddings": 0.05,
            "learning_rate_CNN": 0.05,
            "channel_size": [32, 32, 32, 32, 32, 32],
            "dropout": 0.0,
            "epoch_verbose": 1,
            # hyperparameter from the paper
            "learning_rate": [0.0001, 0.001, 0.01],
            "regularization_coefficient": [0, 0.00001, 0.0001, 0.001, 0.01],
            "dropout_rate": [0, 0.1, 0.3, 0.5, 0.7, 0.9],
            "sampling_size": 50
        }

        # Do not change this
        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

        self._print("Saving complete")

    """
        Loads the model 
    """
    def load_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(folder_path + file_name))

        # Reload the attributes dictionary
        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)

        for attrib_name in data_dict.keys():
            self.__setattr__(attrib_name, data_dict[attrib_name])

        # Done replace this with what required to re-instantiate the model and load its weights,
        #  Call the init_model function you created before

        """
        WE DECIDED NOT TO SAVE ALL THE DICTIONARIES NEEDED FOR THE MODEL ISTANTIATION, BECAUSE THEY ARE REALLY HEAVY
        AND CONTAIN A NUMBER OF WHOLE DATA STRUCTURES. INSTEAD, FOR THE RELOAD, WE LET THE MODEL INITIALIZE ITSELF 
        WITH STANDARD gowalla_config, WHICH IS EXTREMELY FAST. THE TRAINER IS THE SAME FOR ALL THREE CONFIGURATIONS, 
        THE DATASET CONFIG IS NOT NEEDED BECAUSE THE URM WILL GET TRANSFORMED IN THE DATASET DATA STRUCTURE AND PASSED TO get_model,
        WHILE, STRANGELY, ALL THE MODELS ARE THE SAME EXCEPT FOR AMAZON WHICH HAS NO DROPOUT (NO EXPLANATION GIVEN BY THE AUTHORS)
        
        TRADEOFF IS:
        PASS ALL THREE CONFIGS, LET THE WRAPPER USE THEM AGAIN TO INITIALIZE? NO, TOO SLOW
        USE STANDARD GOWALLA CONFIG, JUST EDIT THE MODEL CONFIG AND FEED THE CORRECT DATA STRUCTURE TO OBTAIN THE SAME RESULT? EXTREMELY FAST
        """

        self._init_model()
        self.model.load(folder_path + file_name + "_weights")

        # Done If you are using tensorflow, you may instantiate a new session here
        # Done reset the default graph to "clean" the tensorflow state
        # the version v1 of compact must be used because other versions are deprecated
        #tf.compat.v1.reset_default_graph()
        #saver = tf.compat.v1.train.Saver()
        #saver.restore(self.sess, folder_path + file_name + "_session")

        self._print("Loading complete")
