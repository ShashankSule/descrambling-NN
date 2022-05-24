import matplotlib.pyplot as plt

import numpy as np

import os

import torch

import torch.nn as nn

from torch.utils.data import DataLoader

from torch.utils.data.sampler import SubsetRandomSampler

import pandas as pd











def parameter_swap_initial(predictions):

    Ordered_T_Parameters = []

    Parameters_np = predictions.detach().numpy()[:, :]

    for index_1 in range(predictions.shape[0]):

        Parameter_Set = Parameters_np[index_1]

        c1 = Parameter_Set[0]

        T21 = Parameter_Set[1]

        T22 = Parameter_Set[2]

        if T21 > T22:

            T_sub = T21

            T21 = T22

            T22 = T_sub

            c_sub = 1 - c1

            c1 = c_sub

            Ordered_T_Parameters.append([])

            Ordered_T_Parameters[index_1].append(c1)

            Ordered_T_Parameters[index_1].append(T21)

            Ordered_T_Parameters[index_1].append(T22)

            index_1 += index_1

        else:

            T21 = T21

            T22 = T22

            c1 = c1

            Ordered_T_Parameters.append([])

            Ordered_T_Parameters[index_1].append(c1)

            Ordered_T_Parameters[index_1].append(T21)

            Ordered_T_Parameters[index_1].append(T22)

            index_1 += index_1

    Ordered_T_Parameters_array = np.array(Ordered_T_Parameters)

    predictions = torch.Tensor(Ordered_T_Parameters_array)

    return predictions





def parameter_swap_add_grad(predictions):

    Ordered_T_Parameters = []

    Parameters_np = predictions.detach().numpy()[:, :]

    for index_1 in range(predictions.shape[0]):

        Parameter_Set = Parameters_np[index_1]

        c1 = Parameter_Set[0]

        T21 = Parameter_Set[1]

        T22 = Parameter_Set[2]

        if T21 > T22:

            T_sub = T21

            T21 = T22

            T22 = T_sub

            c_sub = 1 - c1

            c1 = c_sub

            Ordered_T_Parameters.append([])

            Ordered_T_Parameters[index_1].append(c1)

            Ordered_T_Parameters[index_1].append(T21)

            Ordered_T_Parameters[index_1].append(T22)

            index_1 += index_1

        else:

            T21 = T21

            T22 = T22

            c1 = c1

            Ordered_T_Parameters.append([])

            Ordered_T_Parameters[index_1].append(c1)

            Ordered_T_Parameters[index_1].append(T21)

            Ordered_T_Parameters[index_1].append(T22)

            index_1 += index_1

    Ordered_T_Parameters_array = np.array(Ordered_T_Parameters)

    predictions = torch.Tensor([Ordered_T_Parameters_array], requires_grad=True)

    return predictions





def parameter_swap_edited(predictions):

    Ordered_T_Parameters = []

    Parameters_np = predictions.detach().numpy()[:, :]

    for index_1 in range(predictions.shape[0]):

        Parameter_Set = Parameters_np[index_1]

        c1 = Parameter_Set[0]

        T21 = Parameter_Set[1]

        T22 = Parameter_Set[2]

        if T21 > T22:

            T_sub = T21

            T21 = T22

            T22 = T_sub

            c_sub = 1 - c1

            c1 = c_sub

            Ordered_T_Parameters.append([])

            Ordered_T_Parameters[index_1].append(c1)

            Ordered_T_Parameters[index_1].append(T21)

            Ordered_T_Parameters[index_1].append(T22)

            index_1 += index_1

        else:

            T21 = T21

            T22 = T22

            c1 = c1

            Ordered_T_Parameters.append([])

            Ordered_T_Parameters[index_1].append(c1)

            Ordered_T_Parameters[index_1].append(T21)

            Ordered_T_Parameters[index_1].append(T22)

            index_1 += index_1

    Ordered_T_Parameters_array = np.array(Ordered_T_Parameters)

    reordered_values = torch.Tensor(Ordered_T_Parameters_array)

    reordered_values_rows_columns = reordered_values.size()

    x = torch.zeros(reordered_values_rows_columns[0], reordered_values_rows_columns[1], requires_grad=True)

    predictions = x + reordered_values

    return predictions





def parameter_swap_same_tensor(predictions):

    Ordered_T_Parameters = []

    Parameters_np = predictions.detach().numpy()[:, :]

    for index_1 in range(predictions.shape[0]):

        Parameter_Set = Parameters_np[index_1]

        c1 = Parameter_Set[0]

        T21 = Parameter_Set[1]

        T22 = Parameter_Set[2]

        if T21 > T22:

            T_sub = T21

            T21 = T22

            T22 = T_sub

            c_sub = 1 - c1

            c1 = c_sub

            Ordered_T_Parameters.append([])

            Ordered_T_Parameters[index_1].append(c1)

            Ordered_T_Parameters[index_1].append(T21)

            Ordered_T_Parameters[index_1].append(T22)

            index_1 += index_1

        else:

            T21 = T21

            T22 = T22

            c1 = c1

            Ordered_T_Parameters.append([])

            Ordered_T_Parameters[index_1].append(c1)

            Ordered_T_Parameters[index_1].append(T21)

            Ordered_T_Parameters[index_1].append(T22)

            index_1 += index_1

    Ordered_T_Parameters_array = np.array(Ordered_T_Parameters)

    reordered_values = torch.Tensor(Ordered_T_Parameters_array)

    predictions_zeroes = torch.full_like(predictions, 0, requires_grad=True)

    predictions = predictions_zeroes + reordered_values

    return predictions



def parameter_swap_same_tensor_two_variables(predictions):
    Ordered_T_Parameters = []
    predictions.to("cpu")
    Parameters_np = predictions.detach().numpy()[:, :]
    for index_1 in range(predictions.shape[0]):
        Parameter_Set = Parameters_np[index_1]
        T21 = Parameter_Set[0]
        T22 = Parameter_Set[1]
        if T21 > T22:
            T_sub = T21
            T21 = T22
            T22 = T_sub
            Ordered_T_Parameters.append([])
            Ordered_T_Parameters[index_1].append(T21)
            Ordered_T_Parameters[index_1].append(T22)
            index_1 += index_1
        else:
            T21 = T21
            T22 = T22
            Ordered_T_Parameters.append([])
            Ordered_T_Parameters[index_1].append(T21)
            Ordered_T_Parameters[index_1].append(T22)
            index_1 += index_1
    Ordered_T_Parameters_array = np.array(Ordered_T_Parameters)
    reordered_values = torch.Tensor(Ordered_T_Parameters_array)
    predictions_zeroes = torch.full_like(predictions, 0, requires_grad=True)
    predictions = predictions_zeroes + reordered_values
    #Use GPU or CPU?
    if torch.cuda.is_available():
        device = torch.device("cuda:0")#WAS CUDA:7
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    
    predictions.to(device)   #before cuda:6
    return predictions




def parameter_swap_same_tensor_mul_zero(predictions):

    Ordered_T_Parameters = []

    Parameters_np = predictions.detach().numpy()[:, :]

    for index_1 in range(predictions.shape[0]):

        Parameter_Set = Parameters_np[index_1]

        c1 = Parameter_Set[0]

        T21 = Parameter_Set[1]

        T22 = Parameter_Set[2]

        if T21 > T22:

            T_sub = T21

            T21 = T22

            T22 = T_sub

            c_sub = 1 - c1

            c1 = c_sub

            Ordered_T_Parameters.append([])

            Ordered_T_Parameters[index_1].append(c1)

            Ordered_T_Parameters[index_1].append(T21)

            Ordered_T_Parameters[index_1].append(T22)

            index_1 += index_1

        else:

            T21 = T21

            T22 = T22

            c1 = c1

            Ordered_T_Parameters.append([])

            Ordered_T_Parameters[index_1].append(c1)

            Ordered_T_Parameters[index_1].append(T21)

            Ordered_T_Parameters[index_1].append(T22)

            index_1 += index_1

    Ordered_T_Parameters_array = np.array(Ordered_T_Parameters)

    reordered_values = torch.Tensor(Ordered_T_Parameters_array)

    predictions_zeroes = predictions * 0

    predictions = predictions_zeroes + reordered_values

    return predictions



