#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/08/2022

@author: Maurizio Ferrari Dacrema
"""


from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
import scipy.sparse as sps
import numpy as np
import torch, os
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, record_function, ProfilerActivity


def batch_dot(tensor_1, tensor_2):
    """
    Vectorized dot product between rows of two tensors
    :param tensor_1:
    :param tensor_2:
    :return:
    """
    # torch.einsum("ki,ki->k", tensor_1, tensor_2)
    return tensor_1.multiply(tensor_2).sum(axis=1)


class _SimpleMFModel(torch.nn.Module):

    def __init__(self, n_users, n_items, embedding_dim=20):
        super(_SimpleMFModel, self).__init__()

        self._embedding_user = torch.nn.Embedding(n_users, embedding_dim = embedding_dim)
        self._embedding_item = torch.nn.Embedding(n_items, embedding_dim = embedding_dim)


    def forward(self, user, item):
        prediction = batch_dot(self._embedding_user(user), self._embedding_item(item))
        return prediction









class URM_Dataset(Dataset):
    def __init__(self, URM_train):
        super().__init__()
        URM_train = sps.coo_matrix(URM_train)

        self._row = torch.tensor(URM_train.row).type(torch.LongTensor)
        self._col = torch.tensor(URM_train.col).type(torch.LongTensor)
        self._data = torch.tensor(URM_train.data).type(torch.FloatTensor)


    def __len__(self):
        return len(self._row)

    def __getitem__(self, index):
        return self._row[index], self._col[index], self._data[index]


class BPR_Dataset(Dataset):
    def __init__(self, URM_train):
        super().__init__()
        self._URM_train = sps.csr_matrix(URM_train)
        self.n_users, self.n_items = self._URM_train.shape

    def __len__(self):
        return self.n_users

    def __getitem__(self, user_id):

        seen_items = self._URM_train.indices[self._URM_train.indptr[user_id]:self._URM_train.indptr[user_id+1]]
        item_positive = np.random.choice(seen_items)

        # seen_items = set(list(seen_items))
        negative_selected = False

        while not negative_selected:
            negative_candidate = np.random.randint(low=0, high=self.n_items, size=1)[0]

            if negative_candidate not in seen_items:
                item_negative = negative_candidate
                negative_selected = True

        # return torch.tensor(user_id).to("cuda"), torch.tensor(item_positive).to("cuda"), torch.tensor(item_negative).to("cuda")
        return user_id, item_positive, item_negative



def loss_MSE(model, batch):
    user, item, rating = batch

    # Compute prediction for each element in batch
    prediction = model.forward(user, item)

    # Compute total loss for batch
    loss = (prediction - rating).pow(2).mean()

    return loss



def loss_BPR(model, batch):
    user, item_positive, item_negative = batch

    # Compute prediction for each element in batch
    x_ij = model.forward(user, item_positive) - model.forward(user, item_negative)

    # Compute total loss for batch
    loss = -x_ij.sigmoid().log().mean()

    return loss


class _PyTorchMFRecommender(BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping):
    """
    """

    RECOMMENDER_NAME = "_PyTorchMFRecommender"

    def __init__(self, URM_train, verbose = True):
        super(_PyTorchMFRecommender, self).__init__(URM_train, verbose = verbose)

    def fit(self, epochs=300,
            batch_size = 8,
            num_factors = 32,
            l2_reg = 1e-4,
            sgd_mode = 'adam',
            learning_rate = 1e-2,
            **earlystopping_kwargs):

        self._data_loader = DataLoader(self._dataset, batch_size = int(batch_size), shuffle = True, num_workers = os.cpu_count(), pin_memory=True)
        self._model = _SimpleMFModel(self.n_users, self.n_items, embedding_dim = num_factors)
        # self._model.to("cuda")

        if sgd_mode.lower() == "adagrad":
            self._optimizer = torch.optim.Adagrad(self._model.parameters(), lr = learning_rate, weight_decay = l2_reg)
        elif sgd_mode.lower() == "rmsprop":
            self._optimizer = torch.optim.RMSprop(self._model.parameters(), lr = learning_rate, weight_decay = l2_reg)
        elif sgd_mode.lower() == "adam":
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr = learning_rate, weight_decay = l2_reg)
        elif sgd_mode.lower() == "sgd":
            self._optimizer = torch.optim.SGD(self._model.parameters(), lr = learning_rate, weight_decay = l2_reg)
        else:
            raise ValueError("sgd_mode attribute value not recognized.")


        self._update_best_model()

        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:

        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
        # prof.export_chrome_trace("trace.json")

        self._print("Training complete")

        self.USER_factors = self.USER_factors_best.copy()
        self.ITEM_factors = self.ITEM_factors_best.copy()



    def _prepare_model_for_validation(self):
        self.USER_factors = self._model._embedding_user.weight.detach().cpu().numpy()
        self.ITEM_factors = self._model._embedding_item.weight.detach().cpu().numpy()


    def _update_best_model(self):
        self.USER_factors_best = self._model._embedding_user.weight.detach().cpu().numpy()
        self.ITEM_factors_best = self._model._embedding_item.weight.detach().cpu().numpy()


    def _run_epoch(self, num_epoch):

        epoch_loss = 0
        for batch in self._data_loader:

            # Clear previously computed gradients
            self._optimizer.zero_grad()

            loss = self._loss_function(self._model, batch)

            # Compute gradients given current loss
            loss.backward()

            # Apply gradient using the selected optimizer
            self._optimizer.step()

            epoch_loss += loss.item()



class PyTorchMF_BPR_Recommender(_PyTorchMFRecommender):

    RECOMMENDER_NAME = "PyTorchMF_BPR_Recommender"

    def __init__(self, URM_train, verbose = True):
        super(PyTorchMF_BPR_Recommender, self).__init__(URM_train, verbose = verbose)

        self._dataset = BPR_Dataset(self.URM_train)
        self._loss_function = loss_BPR



class PyTorchMF_MSE_Recommender(_PyTorchMFRecommender):

    RECOMMENDER_NAME = "PyTorchMF_MSE_Recommender"

    def __init__(self, URM_train, verbose = True):
        super(PyTorchMF_MSE_Recommender, self).__init__(URM_train, verbose = verbose)

        self._dataset = URM_Dataset(self.URM_train)
        self._loss_function = loss_MSE



