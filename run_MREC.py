#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17
@author: Maurizio Ferrari Dacrema
"""

import os, traceback, argparse
import numpy as np

from Conferences.MREC.MREC_our_interface.DatasetPublic.CityULikeReader import CityULikeReader
from Conferences.MREC.MREC_our_interface.MREC_RecommenderWrapper import MREC_RecommenderWrapper
from Conferences.MREC.MREC_our_interface.DatasetPublic.MovieLens10MReader import *
from Conferences.MREC.MREC_our_interface.DatasetPublic.GowallaReader import *
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchSingleCase import SearchSingleCase
from Utils.ResultFolderLoader import ResultFolderLoader, generate_latex_hyperparameters
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices
from Evaluation.Evaluator import EvaluatorHoldout, EvaluatorNegativeItemSample

import scipy.sparse as sps


def read_data_split_and_search(dataset_name,
                               flag_baselines_tune=False,
                               flag_DL_article_default=False, flag_DL_tune=False,
                               flag_print_results=False):
    # Using dataReader from CollaborativeVAE_our_interface as they use the same data in the same way

    result_folder_path = "result_experiments/{}/{}_MREC_{}/".format(CONFERENCE_NAME, ALGORITHM_NAME,
                                                                            dataset_name)

    if dataset_name == 'ml-10m':
        dataset = MovieLens10MReader("Data_manager_split_datasets/")
    else:
        if dataset_name == 'cite-u-like':
            dataset = CityULikeReader("Data_manager_split_datasets/")
        else:
            if dataset_name == 'gowalla':
                dataset = GowallaReader("Data_manager_split_datasets/")

    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()


    #it will be made implicit by algorithm
    #assert_implicit_data([URM_train, URM_validation, URM_test])

    # Due to the sparsity of the dataset, choosing an evaluation as subset of the train
    #In this case the train data will contain validation data to avoid cold users
    assert_disjoint_matrices([URM_train, URM_test])
    #assert_disjoint_matrices([URM_validation, URM_test])
    URM_train_last_test = URM_train


    metric_to_optimize = 'NDCG'

    cutoff_to_optimize = 10
    cutoff_list=[10.20]
    # If directory does not exist, create
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)

    URM_test = sps.csr_matrix(URM_test)

    evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, exclude_seen=True)

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

    ################################################################################################
    ######
    ######      DL ALGORITHM
    ######

    flag_Dl_article_default = True

    if flag_DL_article_default:

        try:

            article_hyperparameters = {
                'alpha' : 30,
                'K' : 20,
                'max_iter' : 20,
            }

            parameterSearch = SearchSingleCase(MREC_RecommenderWrapper,
                                               evaluator_validation=evaluator_validation,
                                               evaluator_test=evaluator_test)

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
                FIT_KEYWORD_ARGS={'article_hyperparameters':article_hyperparameters})

            recommender_input_args_last_test = recommender_input_args.copy()
            recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test


            parameterSearch.search(recommender_input_args,
                                   recommender_input_args_last_test=recommender_input_args_last_test,
                                   fit_hyperparameters_values={},
                                   cutoff_to_optimize=cutoff_to_optimize,
                                   metric_to_optimize=metric_to_optimize,
                                   output_folder_path=result_folder_path,
                                   resume_from_saved=False,
                                   output_file_name_root=MREC_RecommenderWrapper.RECOMMENDER_NAME,
                                   save_model="best",
                                   evaluate_on_test="best",
                                   )
            # recommender_instance= MREC_RecommenderWrapper(URM_train)
            # recommender_instance.fit()
            # results_df, string= evaluator_test.evaluateRecommender(recommender_instance)
            # print(results_df)


        except Exception as e:

            print("On recommender {} Exception {}".format(MREC_RecommenderWrapper, str(e)))
            traceback.print_exc()

    ################################################################################################
    ######
    ######      PRINT RESULTS
    ######

    if flag_print_results:
        URM_test= sps.csr_matrix(URM_test)
        n_test_users = np.sum(np.ediff1d(URM_test.indptr) >= 1)

        result_loader = ResultFolderLoader(result_folder_path,
                                           base_algorithm_list=None,
                                           other_algorithm_list=None,
                                           KNN_similarity_list=KNN_similarity_to_report_list,
                                           ICM_names_list=dataset.ICM_DICT.keys(),
                                           UCM_names_list=dataset.UCM_DICT.keys(),
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

    ALGORITHM_NAME = "MREC"
    CONFERENCE_NAME = " Improving Implicit Alternating Least Squares with Ring-based Regularization"

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
    dataset_list = ["ml-10m", "cite-u-like", "gowalla", ]

    for dataset_name in dataset_list:
        read_data_split_and_search(dataset_name,
                                   flag_baselines_tune=input_flags.baseline_tune,
                                   flag_DL_article_default=True,
                                   flag_print_results=input_flags.print_results,
                                   )
