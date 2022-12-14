#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from Conferences.RGCF.RGCF_our_interface.DatasetPublic.MovieLens1MReader import Movielens1MReader
from Conferences.RGCF.RGCF_our_interface.DatasetPublic.AmazonReader import AmazonReader
from Conferences.RGCF.RGCF_our_interface.DatasetPublic.YelpReader import YelpReader
from Conferences.RGCF.RGCF_our_interface.RGCF_RecommenderWrapper import RGCF_RecommenderWrapper
from HyperparameterTuning.SearchSingleCase import SearchSingleCase
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.functions_for_parallel_model import _get_model_list_given_dataset, _optimize_single_model
from Recommenders.Recommender_import_list import *
from Utils.ResultFolderLoader import ResultFolderLoader

from functools import partial
import numpy as np
import os, traceback, argparse, multiprocessing

from Conferences.CIKM.ExampleAlgorithm_our_interface.Example_RecommenderWrapper import Example_RecommenderWrapper

from Evaluation.Evaluator import EvaluatorHoldout
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices
from recbole.config import Config
import scipy.sparse as sps

def read_data_split_and_search(dataset_name,
                               flag_baselines_tune=False,
                               flag_DL_article_default=False, flag_DL_tune=False,
                               flag_print_results=False):
    result_folder_path = "result_experiments/{}/{}_{}/".format(CONFERENCE_NAME, ALGORITHM_NAME, dataset_name)
    data_folder_path = result_folder_path + "data/"
    model_folder_path = result_folder_path + "models/"






    """
    All of this stuff below is needed to make recbole work. The authors didn't explain
    their use of recbole, so we tried to emulate what they said (remove less than 15 interaction
    for amazon and yelp) but for some reason some config options are taken
    from internal config, some from final config, but it is not stated which is taken from which,
    so we set both. Option names are self explanatory
    """






    if dataset_name == "ml-1m":


        dataset = Movielens1MReader("Conferences/RGCF/RGCF_github/dataset/")

    elif dataset_name == "yelp2018":

        dataset = YelpReader("Conferences/RGCF/RGCF_github/dataset/")
    elif dataset_name == "Amazon_Books":

        dataset = AmazonReader("Conferences/RGCF/RGCF_github/dataset/Amazon_Books")
    else:
        print("Dataset name not supported, current is {}".format(dataset_name))
        return



    print('Current dataset is: {}'.format(dataset_name))
    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()



    URM_train_last_test = URM_train #Because of recbole data types, it is not possible to merge the matrices.
    ##URM_train cannot be turned back into a recbole data structure, so it will be loaded from file.
    ##It is also not possible to pass it through the datareader as it is not compatible with DataIO

    # Ensure IMPLICIT data and disjoint test-train split
    assert_implicit_data([URM_train, URM_validation, URM_test])
    assert_disjoint_matrices([URM_train, URM_validation, URM_test])

    # If directory does not exist, create
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)

    # Done Replace metric to optimize and cutoffs
    metric_to_optimize = 'MRR'
    cutoff_to_optimize = 10

    # All cutoffs that will be evaluated are listed here
    cutoff_list = [5, 10, 20, 30, 40, 50, 100]
    max_total_time = 14 * 24 * 60 * 60  # 14 days

    n_cases = 50
    n_processes = 3
    resume_from_saved = False

    # Done Select the evaluation protocol
    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=cutoff_list)
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    evaluator_validation_earlystopping = EvaluatorHoldout(URM_validation, cutoff_list=[cutoff_to_optimize])

    ################################################################################################
    ######
    ######      DL ALGORITHM
    ######

    if flag_DL_article_default:
        try:
            # Done fill this dictionary with the hyperparameters of the algorithm
            article_hyperparameters = {
                # modified
                #added
                "dataset_name":dataset_name,
                "number_of_layers_K": [2, 3],
                "learning_rate": [1e-5, 1e-3],
                "pruning_threshold_beta":[0.02, 0.04, 0.1],
                "temperature_tau": [0.05, 0.1, 0.2],
                "diversity_loss_coefficient_lambda1": [1e-5, 1e-6, 1e-7, 1e-8],
                "regularization_coefficient_lambda2": 1e-5,
                #default
                "epochs_MFBPR": 500,
                "embedding_size": 64,
                "hidden_size": 128,
                "negative_instances_per_positive": 4,
                "regularization_users_items": 0.01,
                "regularization_weights": 10,
                "regularization_filter_weights": 1,
                "learning_rate_embeddings": 0.05,
                "learning_rate_CNN": 0.05,
                "channel_size": [32, 32, 32, 32, 32, 32],
                "dropout": 0.0,
                "epoch_verbose": 1,
                ##from code
                "topk": [10, 20, 50],
                "negative_sample_per_positive": 1,
                "epochs": 500,
                "batch_size": 4096,

            }

            # Do not modify earlystopping
            earlystopping_hyperparameters = {"validation_every_n": 1,
                                             "stop_on_validation": True,
                                             "lower_validations_allowed": 10,
                                             "evaluator_object": evaluator_validation,
                                             "validation_metric": metric_to_optimize,
                                             }


            #
            """
            To load the correct DataLoader, an object previously
            saved from DataReader, the wrapper will need to read from a file.
            For now, we chose to pass that file name as an hyperparameter.
            It would have been good to pass it directly from the DataReader
            to the Wrapper, but unluckily TrainDataLoader is not compatible with DataIO. Reading
            from file is the only solution remaining. To make it work dataset_name should be saved
            """


            # Fit the DL model, select the optimal number of epochs and save the result
            hyperparameterSearch = SearchSingleCase(RGCF_RecommenderWrapper,
                                                    evaluator_validation=evaluator_validation,
                                                    evaluator_test=evaluator_test)

            # Specify which attributes are needed. In this case the constructor only required the URM train,
            # no additional fit arguments are required (besides those that are listed previously in the hyperparameters dictionary)
            # and the hyperparameters required by the earlystopping are provided separately.
            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
                FIT_KEYWORD_ARGS={"article_hyperparameters":article_hyperparameters},
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
                                        output_file_name_root=Example_RecommenderWrapper.RECOMMENDER_NAME,
                                        resume_from_saved=resume_from_saved,
                                        save_model="best",
                                        evaluate_on_test="best",
                                        )



        except Exception as e:

            print("On recommender {} Exception {}".format(Example_RecommenderWrapper, str(e)))
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
                                                         dataset.ICM_DICT,
                                                         dataset.UCM_DICT)

        _optimize_single_model_partial = partial(_optimize_single_model,
                                                 URM_train=URM_train,
                                                 URM_train_last_test=URM_train_last_test,
                                                 n_cases=n_cases,
                                                 n_random_starts=int(n_cases / 3),
                                                 resume_from_saved=True,
                                                 save_model="best",
                                                 evaluate_on_test="best",
                                                 evaluator_validation=evaluator_validation,
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
        URM_test=sps.csr_matrix(URM_train)

        n_test_users = np.sum(np.ediff1d(URM_test.indptr) >= 1)

        result_loader = ResultFolderLoader(model_folder_path,
                                           base_algorithm_list=None,
                                           other_algorithm_list=None,
                                           KNN_similarity_list=KNN_similarity_to_report_list,
                                           ICM_names_list=dataset.ICM_DICT.keys(),
                                           UCM_names_list=dataset.UCM_DICT.keys(),
                                           )

        result_loader.generate_latex_results(result_folder_path + "{}_latex_results.txt".format("accuracy_metrics"),
                                             metrics_list=['RECALL', 'PRECISION', 'MAP', 'NDCG','MRR'],
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
    ALGORITHM_NAME = "RGCF"
    CONFERENCE_NAME = "Learning to Denoise Unreliable Interactions for Graph Collaborative Filtering"

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
    ##[!] WE COULDN'T UNDERSTAND WHICH VERSION OF YELP IT IS
    ##atomic file dataset are all available here https://drive.google.com/drive/folders/1so0lckI6N6_niVEYaBu-LIcpOdZf99kj
    dataset_list = ["ml-1m-main","Amazon_Books","yelp2018",]

    for dataset_name in dataset_list:
        read_data_split_and_search(dataset_name,
                                   flag_baselines_tune=input_flags.baseline_tune,
                                   flag_DL_article_default=input_flags.DL_article_default,
                                   flag_print_results=input_flags.print_results,
                                   )

