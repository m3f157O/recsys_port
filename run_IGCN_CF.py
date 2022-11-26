#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from Conferences.IGCN_CF.IGCN_CF_RecommenderWrapper import IGCN_CF_RecommenderWrapper
from Conferences.IGCN_CF.igcn_cf_github.config import *
from Conferences.IGCN_CF.igcn_cf_our_interface.DatasetPublic.GowallaReader import GowallaReader
from Conferences.IGCN_CF.igcn_cf_our_interface.DatasetPublic.AmazonReader import AmazonReader
from Conferences.IGCN_CF.igcn_cf_our_interface.DatasetPublic.IGN_CFReader import init_file_and_device, acquire_dataset
from Conferences.IGCN_CF.igcn_cf_our_interface.DatasetPublic.YelpReader import YelpReader
from HyperparameterTuning.SearchSingleCase import SearchSingleCase
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.functions_for_parallel_model import _get_model_list_given_dataset, _optimize_single_model
from Recommenders.Recommender_import_list import *
from Utils.ResultFolderLoader import ResultFolderLoader
import scipy.sparse as sps
from Conferences.IGCN_CF.igcn_cf_github.utils import init_run

import torch

from functools import partial
import numpy as np
import os, traceback, argparse, multiprocessing

from Evaluation.Evaluator import EvaluatorHoldout, EvaluatorNegativeItemSample
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices
from Conferences.IGCN_CF.igcn_cf_our_interface.DatasetPublic.IGN_CFReader import adjacencyList2COO,init_file_and_device,acquire_dataset,preprocessing

""""
    Class to build the dataset with the sparse matrices taken by the IGN_CFReader
"""

""""
    Convert a sparse matrix to an adjacent list
"""



def read_data_split_and_search(dataset_name,
                               flag_baselines_tune=False,
                               flag_DL_article_default=False, flag_DL_tune=False,
                               flag_print_results=False):
    result_folder_path = "result_experiments/{}/{}_{}/".format(CONFERENCE_NAME, ALGORITHM_NAME, dataset_name)
    data_folder_path = result_folder_path + "data/"
    model_folder_path = result_folder_path + "models/"

    # Done: Replace with dataset name and relative DataReader
    #  The two datareaders correspond to two examples, CiteULike as an example of dataset provided int the original repository
    #  while Movielens20M as a dataset not provided in the repository but publicly available, in that case one of the readers
    #  already available in this repository could be used



    device=torch.device('cuda')


    ##The configuration is necessary for the reader,
    ##to use the original reading function (model.get_model())

    ##Moreover, the dataset is a custom object, which needs to inherit
    ##from BasicDataset class. We made a trick and bypassed this limitation,
    ##but the config is necessary anyway

    base_path="Conferences/IGCN_CF/igcn_cf_our_interface/DatasetPublic/data/"
    if dataset_name == "yelp":
        config = get_yelp_config(device)
        dataset_config, model_config, trainer_config = config[2]
        pre_splitted_path = base_path+"Yelp/"
        dataset_reader = YelpReader(pre_splitted_path)
    elif dataset_name == "amazon-book":
        config = get_amazon_config(device)
        dataset_config, model_config, trainer_config = config[2]
        pre_splitted_path = base_path+"Amazon/"
        dataset_reader = AmazonReader(pre_splitted_path)
    elif dataset_name == "gowalla":
        config = get_gowalla_config(device)
        dataset_config, model_config, trainer_config = config[2]
        pre_splitted_path = base_path+"Gowalla/"
        dataset_reader = GowallaReader(pre_splitted_path,config)
    else:
        print("Dataset name not supported, current is {}".format(dataset_name))
        return





    URM_train = dataset_reader.URM_DICT["URM_train"].copy()
    URM_validation = dataset_reader.URM_DICT["URM_validation"].copy()
    URM_test = dataset_reader.URM_DICT["URM_test"].copy()

    URM_train_last_test = URM_train + URM_validation

    # Ensure IMPLICIT data and disjoint test-train split
    assert_implicit_data([URM_train, URM_validation, URM_test])
    assert_disjoint_matrices([URM_train, URM_validation, URM_test])

    # If directory does not exist, create
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)

    # Done Replace metric to optimize and cutoffs
    metric_to_optimize = 'NDCG'
    # when the early stopping must terminate
    cutoff_to_optimize = 20

    # All cutoffs that will be evaluated are listed here
    cutoff_list = [5, 10, 20, 30, 40, 50, 100]
    max_total_time = 14 * 24 * 60 * 60  # 14 days

    # parameters taken from the articles section 5.1
    n_cases = 20
    n_processes = 5
    resume_from_saved = True

    # Done Select the evaluation protocol
    evaluator_validation_earlystopping = EvaluatorHoldout(URM_validation, cutoff_list=[cutoff_to_optimize])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

    ################################################################################################
    ######
    ######      DL ALGORITHM
    ######

    """
        dataset config for gowalla     dataset_config = {'name': 'ProcessedDataset', 'path': 'data/Gowalla/time','device': torch.device('cuda')}
        model config for gowalla         model_config = {'name': 'IGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                                                        'dropout': 0.3, 'feature_ratio': 1.}
        trainer config for gowalla       trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'aux_reg': 0.01,
                                        'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                                        'test_batch_size': 512, 'topks': [20]}
        IN GENERAL, THE MODEL CONFIG DOESN'T DIFFER BETWEEN DATASETS, EXCEPT FOR AMAZON, WHERE
        THERE IS NO DROPOUT AT ALL. THIS MAY BE A PROBLEM WHEN THE MODEL IS RELOADED, BECAUSE,
        GIVEN THAT THE ENTIRE DATASET STRUCTURE IS NEEDED TO CALL get_model(), TO AVOID
        SAVING AND RELOADING AND PASSING THE configs TO init_model() IN IGCN_CF_RecommenderWrapper.py,
        WHICH ARE !!!REALLY!!! HEAVY DATA STRUCTURES (they contain the reference to a number of objects used during training)
        WE DECIDED TO ADD A DEFAULT LOADING OPTION IN init_model, WHICH 
        WILL ALWAYS PROCESS THE DATASET CORRECTLY, AND LOAD THE CORRECT TRAINER, 
        BUT THE STANDARD MODEL CONFIG (gowalla) BE WILL MODIFIED ON THE ['dropout'] ENTRY
        
        THIS IS AN "HARDCODED" SOLUTION TO SPEED UP MODEL INSTANTIATION WHEN RELOADING. IF IT'S A BAD PRACTICE, 
        OR AGAINST THE REQUIREMENTS, IT WOULD BE EASY FOR US TO CORRECT IT

    """

    if flag_DL_article_default:

        try:

            """
            WE COULD HAVE "UNPACKED" THE CONFIGS IN A MORE CLEAN WAY,
            BUT SHORTLY AFTER WE WOULD HAVE HAD TO REPACK THEM, RESULTING
            IN A LOT MORE CODE
            """
            # Done fill this dictionary with the hyperparameters of the algorithm
            article_hyperparameters = {
                  "dataset_config":dataset_config,
                  "model_config": model_config,
                  "trainer_config": trainer_config,
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
                  "learning_rate_vae":1e-2,
                  "learning_rate_cvae":1e-3,
                  "num_factors":50,
                  "dimensions_vae":[200, 100],
                  "epochs_vae":[50, 50],
                  "lambda_u":0.1,
                  "lambda_v":10,
                  "lambda_r":1,
                  "M":300,
                  "learning_rate": [0.0001, 0.001, 0.01],
                  "regularization_coefficient": [0, 0.00001, 0.0001, 0.001, 0.01],
                  "dropout_rate": [0, 0.1, 0.3, 0.5, 0.7, 0.9],
                  "sampling_size":50,
                  "batch_size":2048,
                  "a":1,
                  "b":0.01,
                  "epochs":1000,
                  "embedding_size":64,
                  "temp_file_folder":None,

            }

            # Do not modify earlystopping
            earlystopping_hyperparameters = {"validation_every_n": 5,
                                             "stop_on_validation": True,
                                             "lower_validations_allowed": 5,
                                             "evaluator_object": evaluator_validation_earlystopping,
                                             "validation_metric": metric_to_optimize,
                                             }





            # This is a simple version of the tuning code that is reported below and uses SearchSingleCase
            # You may use this for a simpler testing
            recommender_instance = IGCN_CF_RecommenderWrapper(URM_train)




            recommender_instance.fit(article_hyperparameters,**earlystopping_hyperparameters)
            #
            # recommender_instance.fit(**article_hyperparameters,
            #                          **earlystopping_hyperparameters)
            #
            # evaluator_test.evaluateRecommender(recommender_instance)

            # Fit the DL model, select the optimal number of epochs and save the result
            hyperparameterSearch = SearchSingleCase(IGCN_CF_RecommenderWrapper,
                                                    evaluator_validation=evaluator_validation_earlystopping,
                                                    evaluator_test=evaluator_test)

            # Specify which attributes are needed. In this case the constructor only required the URM train,
            # no additional fit arguments are required (besides those that are listed previously in the hyperparameters dictionary)
            # and the hyperparameters required by the earlystopping are provided separately.
            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
                FIT_KEYWORD_ARGS={},
                EARLYSTOPPING_KEYWORD_ARGS=earlystopping_hyperparameters)

            # Create the attributes needed to fit the last model on the union of training and validation data
            # This model will be fit with the optimal hyperparameters found and then will be evaluated on the test data
            recommender_input_args_last_test = recommender_input_args.copy()
            recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test

            hyperparameterSearch.search(recommender_input_args,
                                        recommender_input_args_last_test=recommender_input_args_last_test,
                                        fit_hyperparameters_values={},
                                        metric_to_optimize=metric_to_optimize,
                                        cutoff_to_optimize=cutoff_to_optimize,
                                        output_folder_path=model_folder_path,
                                        output_file_name_root=IGCN_CF_RecommenderWrapper.RECOMMENDER_NAME,
                                        resume_from_saved=resume_from_saved,
                                        save_model="best",
                                        evaluate_on_test="best",
                                        )



        except Exception as e:

            print("On recommender {} Exception {}".format(IGCN_CF_RecommenderWrapper, str(e)))
            traceback.print_exc()

    ################################################################################################
    ######
    ######      BASELINE ALGORITHMS - Nothing should be modified below this point
    ######

    if flag_baselines_tune:
        recommender_class_list = [
            Random,
            TopPop,
            GlobalEffects,
            # SLIMElasticNetRecommender,
            # UserKNNCFRecommender,
            # MatrixFactorization_BPR_Cython,
            # IALSRecommender,
            # MatrixFactorization_FunkSVD_Cython,
            # EASE_R_Recommender,
            ItemKNNCFRecommender,
            # P3alphaRecommender,
            # SLIM_BPR_Cython,
            # RP3betaRecommender,
            PureSVDRecommender,
            # NMFRecommender,
            # UserKNNCBFRecommender,
            # ItemKNNCBFRecommender,
            # UserKNN_CFCBF_Hybrid_Recommender,
            # ItemKNN_CFCBF_Hybrid_Recommender,
            # LightFMCFRecommender,
            # LightFMUserHybridRecommender,
            # LightFMItemHybridRecommender,
            # MultVAERecommender,
        ]

        model_cases_list = _get_model_list_given_dataset(recommender_class_list, KNN_similarity_to_report_list,
                                                         dataset_reader.ICM_DICT,
                                                         dataset_reader.UCM_DICT)

        _optimize_single_model_partial = partial(_optimize_single_model,
                                                 URM_train=URM_train,
                                                 URM_train_last_test=URM_train_last_test,
                                                 n_cases=n_cases,
                                                 n_random_starts=int(n_cases / 3),
                                                 resume_from_saved=True,
                                                 save_model="best",
                                                 evaluate_on_test="best",
                                                 evaluator_validation=evaluator_validation_earlystopping,
                                                 evaluator_test=evaluator_test,
                                                 max_total_time=max_total_time,
                                                 evaluator_validation_earlystopping=evaluator_validation_earlystopping,
                                                 metric_to_optimize=metric_to_optimize,
                                                 cutoff_to_optimize=cutoff_to_optimize,
                                                 model_folder_path=model_folder_path)

        pool = multiprocessing.Pool(processes=n_processes, maxtasksperchild=1)
        resultList = pool.map(_optimize_single_model_partial, model_cases_list, chunksize=1)

        pool.close()
        pool.join()

    ################################################################################################
    ######
    ######      PRINT RESULTS
    ######

    if flag_print_results:
        URM_test= sps.csr_matrix(URM_test)
        n_test_users = np.sum(np.ediff1d(URM_test.indptr) >= 1)

        result_loader = ResultFolderLoader(model_folder_path,
                                           base_algorithm_list=None,
                                           other_algorithm_list=None,
                                           KNN_similarity_list=KNN_similarity_to_report_list,
                                           ICM_names_list=dataset_reader.ICM_DICT.keys(),
                                           UCM_names_list=dataset_reader.UCM_DICT.keys(),
                                           )

        result_loader.generate_latex_results(result_folder_path + "{}_latex_results.txt".format("accuracy_metrics"),
                                             metrics_list=['RECALL', 'PRECISION', 'MAP', 'NDCG'],
                                             cutoffs_list=[cutoff_to_optimize],
                                             table_title=None,
                                             highlight_best=True)

        result_loader.generate_latex_results(
            result_folder_path + "{}_latex_results.txt".format("beyond_accuracy_metrics"),
            metrics_list=["NOVELTY", "DIVERSITY_MEAN_INTER_LIST", "COVERAGE_ITEM", "DIVERSITY_GINI", "SHANNON_ENTROPY"],
            cutoffs_list=cutoff_list,
            table_title=None,
            highlight_best=True)

        result_loader.generate_latex_time_statistics(result_folder_path + "{}_latex_results.txt".format("time"),
                                                     n_evaluation_users=n_test_users,
                                                     table_title=None)


if __name__ == '__main__':

    # Done: Replace with algorithm and conference name
    ALGORITHM_NAME = "IGCN_CF"
    CONFERENCE_NAME = "INMO: A Model-Agnostic and Scalable Module for Inductive Collaborative Filtering"

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--baseline_tune', help="Baseline hyperparameter search", type=bool, default=True)
    parser.add_argument('-a', '--DL_article_default', help="Train the DL model with article hyperparameters", type=bool,
                        default=True)
    parser.add_argument('-p', '--print_results', help="Print results", type=bool, default=True)

    input_flags = parser.parse_args()
    print(input_flags)

    # Reporting only the cosine similarity is enough
    KNN_similarity_to_report_list = ["cosine"]  # , "dice", "jaccard", "asymmetric", "tversky"]

    # Done: Replace with dataset names
    dataset_list = ["gowalla"]

    for dataset_name in dataset_list:
        read_data_split_and_search(dataset_name,
                                   flag_baselines_tune=input_flags.baseline_tune,
                                   flag_DL_article_default=input_flags.DL_article_default,
                                   flag_print_results=input_flags.print_results,
                                   )
