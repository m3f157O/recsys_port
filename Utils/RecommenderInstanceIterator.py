"""
Created on 27/09/2019

@author: Maurizio Ferrari Dacrema
"""


from Recommenders.Recommender_import_list import *
from Recommenders.BaseCBFRecommender import BaseItemCBFRecommender, BaseUserCBFRecommender
from Recommenders.FactorizationMachines.LightFMRecommender import _BaseLightFMWrapper

class RecommenderInstanceIterator(object):
    """RecommenderInstanceIterator"""

    def __init__(self,
                 recommender_class_list = None,
                 KNN_similarity_list = None,
                 URM = None,
                 ICM_dict = None,
                 UCM_dict = None,
                 ):

        assert URM is not None, "RecommenderInstanceIterator: Model requires URM, which is None."

        self._recommender_class_list = recommender_class_list
        self._recommender_instance_list = []
        self._recommender_name_list = []
        self._name_to_instance_dict = {}
        self._KNN_similarity_list = KNN_similarity_list
        self._URM = URM
        self._ICM_dict = ICM_dict
        self._UCM_dict = UCM_dict
        self._build_instance_list()


    # def get_recommender_instance(self, recommender_class):
    #
    #     constructor_args_dict = {"URM": self._URM}
    #
    #     if recommender_class is ItemKNNCBFRecommender:
    #         assert self._ICM is not None, "RecommenderFactory: Model requires ICM, which is None."
    #         constructor_args_dict["ICM"] = self._ICM
    #
    #     elif recommender_class is UserKNNCBFRecommender:
    #         assert self._UCM is not None, "RecommenderFactory: Model requires UCM, which is None."
    #         constructor_args_dict["UCM"] = self._UCM
    #
    #     # Allow to add a custom function?
    #
    #     recommender_instance = recommender_class(**constructor_args_dict)
    #
    #     return recommender_instance

    def get_instance_from_name(self, recommender_name):
        return self._name_to_instance_dict[recommender_name]


    def _build_instance_list(self):

        self._recommender_instance_list = []
        self._recommender_name_list = []
        self._name_to_instance_dict = {}

        for recommender_class in self._recommender_class_list:

            if issubclass(recommender_class, BaseItemCBFRecommender):

                for ICM_name, ICM_object in self._ICM_dict.items():
                    recommender_instance = recommender_class(self._URM, ICM_object)
                    # ICM_label = "_{}".format(ICM_name) if len(self._ICM_dict)>1 else ""
                    ICM_label = "_{}".format(ICM_name)

                    # For some algorithms, FactorizationMachines, there is one model per ICM
                    if issubclass(recommender_class, _BaseLightFMWrapper):
                        recommender_name = recommender_instance.RECOMMENDER_NAME + ICM_label

                        self._recommender_instance_list.append(recommender_instance)
                        self._recommender_name_list.append(recommender_name)
                        self._name_to_instance_dict[recommender_name] = recommender_instance

                    # For other algorithms, KNN, there is a model per each similarity heuristic
                    else:
                        for KNN_similarity in self._KNN_similarity_list:
                            recommender_name = recommender_instance.RECOMMENDER_NAME + ICM_label + "_" + KNN_similarity

                            self._recommender_instance_list.append(recommender_instance)
                            self._recommender_name_list.append(recommender_name)
                            self._name_to_instance_dict[recommender_name] = recommender_instance

            elif issubclass(recommender_class, BaseUserCBFRecommender):

                for UCM_name, UCM_object in self._UCM_dict.items():
                    recommender_instance = recommender_class(self._URM, UCM_object)
                    # UCM_label = "_{}".format(UCM_name) if len(self._UCM_dict) > 1 else ""
                    UCM_label = "_{}".format(UCM_name)

                    # For some algorithms, FactorizationMachines, there is one model per UCM
                    if issubclass(recommender_class, _BaseLightFMWrapper):
                        recommender_name = recommender_instance.RECOMMENDER_NAME + UCM_label

                        self._recommender_instance_list.append(recommender_instance)
                        self._recommender_name_list.append(recommender_name)
                        self._name_to_instance_dict[recommender_name] = recommender_instance

                    # For other algorithms, KNN, there is a model per each similarity heuristic
                    else:
                        for KNN_similarity in self._KNN_similarity_list:
                            recommender_name = recommender_instance.RECOMMENDER_NAME + UCM_label + "_" + KNN_similarity

                            self._recommender_instance_list.append(recommender_instance)
                            self._recommender_name_list.append(recommender_name)
                            self._name_to_instance_dict[recommender_name] = recommender_instance

            else:
                recommender_instance = recommender_class(self._URM)

                if recommender_class in [ItemKNNCFRecommender, UserKNNCFRecommender]:

                    for KNN_similarity in self._KNN_similarity_list:
                        recommender_name = recommender_instance.RECOMMENDER_NAME + "_" + KNN_similarity

                        self._recommender_instance_list.append(recommender_instance)
                        self._recommender_name_list.append(recommender_name)
                        self._name_to_instance_dict[recommender_name] = recommender_instance

                else:
                    recommender_instance = recommender_class(self._URM)
                    recommender_name = recommender_instance.RECOMMENDER_NAME
                    self._recommender_instance_list.append(recommender_instance)
                    self._recommender_name_list.append(recommender_name)
                    self._name_to_instance_dict[recommender_name] = recommender_instance



            assert len(self._recommender_instance_list) == len(self._recommender_name_list), "List of instances and names do not have the same length"



    def __iter__(self):
        self._instance_index = 0
        return self

    def __next__(self):
        if self._instance_index < len(self._recommender_name_list):
            recommender_instance = self._recommender_instance_list[self._instance_index]
            recommender_name = self._recommender_name_list[self._instance_index]
            self._instance_index += 1
            return recommender_instance, recommender_name
        else:
            raise StopIteration

    def len(self):
        return len(self._recommender_instance_list)

    def rewind(self):
        self._instance_index = 0




class RecommenderInstanceIterator_Loaded(RecommenderInstanceIterator):
    """RecommenderInstanceIterator"""

    def __init__(self,
                 recommender_class_list = None,
                 KNN_similarity_list = None,
                 URM = None,
                 ICM_dict = None,
                 UCM_dict = None,
                 model_folder_path = None,
                 saved_model_suffix = None
                 ):
        super(RecommenderInstanceIterator_Loaded, self).__init__(recommender_class_list = recommender_class_list,
                                                                 KNN_similarity_list = KNN_similarity_list,
                                                                 URM = URM,
                                                                 ICM_dict = ICM_dict,
                                                                 UCM_dict = UCM_dict)


        for index, recommender_instance in enumerate(self._recommender_instance_list):
            try:
                recommender_instance.load_model(model_folder_path, file_name = self._recommender_name_list[index] + saved_model_suffix)
            except:
                self._recommender_instance_list[index] = None
                self._name_to_instance_dict[self._recommender_name_list[index]] = None


class RecommenderInstanceIterator_Light(object):
    """RecommenderInstanceIterator"""

    def __init__(self,
                 recommender_class_list = None,
                 KNN_similarity_list = None,
                 URM = None,
                 ICM_dict = None,
                 UCM_dict = None,
                 ):

        assert URM is not None, "RecommenderInstanceIterator: Model requires URM, which is None."

        self._recommender_class_list = recommender_class_list
        self._KNN_similarity_list = KNN_similarity_list
        self._URM = URM
        self._ICM_dict = ICM_dict
        self._UCM_dict = UCM_dict

    def __iter__(self):

        for recommender_class in self._recommender_class_list:

            if issubclass(recommender_class, BaseItemCBFRecommender):

                for ICM_name, ICM_object in self._ICM_dict.items():
                    recommender_instance = recommender_class(self._URM, ICM_object)
                    ICM_label = "_{}".format(ICM_name) if len(self._ICM_dict)>1 else ""

                    for KNN_similarity in self._KNN_similarity_list:
                        recommender_name = recommender_instance.RECOMMENDER_NAME + ICM_label + "_" + KNN_similarity

                        yield recommender_instance, recommender_name

            elif issubclass(recommender_class, BaseUserCBFRecommender):

                for UCM_name, UCM_object in self._UCM_dict.items():
                    recommender_instance = recommender_class(self._URM, UCM_object)
                    UCM_label = "_{}".format(UCM_name) if len(self._UCM_dict) > 1 else ""

                    for KNN_similarity in self._KNN_similarity_list:
                        recommender_name = recommender_instance.RECOMMENDER_NAME + UCM_label + "_" + KNN_similarity

                        yield recommender_instance, recommender_name

            else:
                recommender_instance = recommender_class(self._URM)

                if recommender_class in [ItemKNNCFRecommender, UserKNNCFRecommender]:

                    for KNN_similarity in self._KNN_similarity_list:
                        recommender_instance = recommender_class(self._URM)
                        recommender_name = recommender_instance.RECOMMENDER_NAME + "_" + KNN_similarity

                        yield recommender_instance, recommender_name

                else:
                    recommender_instance = recommender_class(self._URM)
                    recommender_name = recommender_instance.RECOMMENDER_NAME

                    yield recommender_instance, recommender_name









