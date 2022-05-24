from makeSignals import myTrueModel, myTrueModel_2param

import matplotlib.pyplot as plt

import numpy as np

import torch

import random

from torch.utils.data import Dataset

from typing import List













class NoisyDecaysDataSet(Dataset):



    def __init__(self, csv_file, target_names, time_bounds, noisy_concat=1):

        """

        Input:

        ------

        1. csv_file (String): Path to the csv file containing the data set. This 

            is a saved DataFrame.

        2. target_names (Array of strings): The names of the target values the NN

            aims to predict. These are keys in the DataFrame saved to the csv

            file that csv_file points to.

        3. time_bounds (Array of floats): The time of the first measurement of

            the signal and the time of the last measurement.

        4. noisy_concat (integer): When an integer greater than 1 is inputted, the time series will

            be duplicated by the input number and concatenated. When the integer is

            equal to 1, no duplication or concatenation will occur.

        """

        self.master_frame = torch.load(csv_file)

        self.training_frame = self.master_frame['noisy_decay']

        self.target_frame   = self.master_frame[target_names]



        # number of times at which the noisy decay is sampled

        self.num_times = len(self.master_frame['noisy_decay'][0])

        # exact times at which the noisy decay is sampled

        self.times = torch.linspace(time_bounds[0],

                                 time_bounds[1],

                                 self.num_times,

                                dtype=torch.float64)

        self.noisy_concat = noisy_concat

        

    def __len__(self):

        """

        Returns the number of samples in the data set

        """

        return len(self.master_frame)



    def __getitem__(self, idx):

        """

        Returns sample(s) at idx. 

        - If idx is an int, the output is a pair of 1-d Tensors: the first is 

        the noisy decay of the sample at the given index  and the second is the

        corresponding target parameters.

        - If idx is a list, it returns two tensors as before whose first

        index points to the particular sample and second index points to a value.

        """

        num_targets = len(self.target_frame.columns)

        if self.noisy_concat == 1:
            noisy_decay = torch.empty(self.num_times, dtype=torch.float64)

            noisy_decay = self.training_frame[idx]

        if self.noisy_concat > 1:
            noisy_decay = torch.empty(self.noisy_concat * self.num_times, dtype=torch.float64)

            noisy_decay = torch.cat([self.training_frame[idx]] * self.noisy_concat)

        target = torch.empty(num_targets, dtype=torch.float64)

        target = torch.tensor([self.target_frame.iloc[idx, feature]

                               for feature in range(num_targets)],

                              dtype=torch.double)


        #print(noisy_decay)

        return noisy_decay, target


class Noisy_RegularizedDecaysDataSet(Dataset):

    def __init__(self, csv_file, target_names, time_bounds):

        """

        Input:

        ------

        1. csv_file (string): Path to the csv file containing the data set. This

            is a saved DataFrame.

        2. target_names (list of strings): The names of the target values the NN

            aims to predict. These are keys in the DataFrame saved to the csv

            file that csv_file points to.

        3. time_bounds (list of 2 floats): Bounds on the times the decay is

            measured at.  The lower bound is time_bounds[0] and the upper bound

            is time_bounds[1].

        """

        self.master_frame = torch.load(csv_file)

        # triples (c1_lambda,T21_lambda,T22_lambda) used to define regularized...

        # ...decays in the training data

        self.training_trios_frame = self.master_frame[['c1_traj',

                                                       'T21_traj',

                                                       'T22_traj']]

        self.target_frame = self.master_frame[target_names]

        # number of regularization parameters and NLLS solution

        self.traj_len = len(self.master_frame['reg_params'][0]) + 1

        self.num_times = len(self.master_frame['noisy_decay'][0])

        self.times = torch.linspace(time_bounds[0],

                                    time_bounds[1],

                                    self.num_times,

                                    dtype=torch.float64)

    def __len__(self):

        """

        Returns the number of samples in the data set

        """

        return len(self.master_frame)

    def __getitem__(self, idx):

        """

        Returns samples at idx.

        - If idx is an int, the output is a pair of 1-d Tensors: the first is

        the regularized decays stacked horizontally and the second is the

        corresponding target parameters.

        - If idx is a list, it returns two Tensors, each with 2 dimensions.

        The first index points to the particular sample and the second index

        points to an intensity at some time for some regularized decay.

        """

        num_targets = len(self.target_frame.columns)

        # number of elements in 1 concatenated regularized decay

        concat_len = self.traj_len * self.num_times

        # the regularized NLLS solutions (c1_ld, T21_ld, T22_ld) for the given sample idx

        regularized_triples = zip(self.training_trios_frame['c1_traj'][idx],

                                  self.training_trios_frame['T21_traj'][idx],

                                  self.training_trios_frame['T22_traj'][idx])

        noisy_regularized_decays = torch.empty(concat_len, dtype=torch.float64)

        # gives the noisy decay first

        noisy_regularized_decays[0:self.num_times] = self.master_frame['noisy_decay'][idx]

        # gives the regularized decays

        noisy_regularized_decays[self.num_times:concat_len] = torch.cat([myTrueModel_2param(self.times,

                                                                                            c1,

                                                                                            1.0 - c1,

                                                                                            T21,

                                                                                            T22,

                                                                                            signalType='biexponential')

                                                                         for c1, T21, T22 in regularized_triples])
        target = torch.empty(num_targets, dtype=torch.float64)

        target = torch.tensor([self.target_frame.iloc[idx, feature]

                               for feature in range(num_targets)],

                              dtype=torch.double)

        # print(regularized_decays)
        return noisy_regularized_decays, target










#===============================================================================#



























class RegularizationTrajectoryDataSet(Dataset):



    def __init__(self, csv_file, particular_trajectories, target_names, time_bounds):

        """

        Input:

        ------

        1. csv_file (string): Path to the csv file containing the data set. This 

            is a saved DataFrame.

        2. particular_trajectories (Array of strings): The names of the particular

            regularization trajectories to be included in the data. These are 

            keys in the DataFrame saved to the csv file that csv_file points to.

        3. target_names (Array of strings): The names of the target values the NN

            aims to predict. These are keys in the DataFrame saved to the csv

            file that csv_file points to.

        """

        self.master_frame = torch.load(csv_file)

        self.training_frame = self.master_frame[particular_trajectories]

        self.target_frame   = self.master_frame[target_names]

        self.traj_len = len(self.master_frame['reg_params'][0])

        self.num_times = len(self.master_frame['noisy_decay'][0])

        self.times = torch.linspace(time_bounds[0],

                                 time_bounds[1],

                                 self.num_times)



        

    def __len__(self):

        """

        Returns the number of samples in the data set

        """

        return len(self.master_frame)



    def __getitem__(self, idx):

        """

        Returns samples at idx. If idx is an int, the output is a pair of

        1-d Tensors: the first is the regularization trajectories stacked

        horizontally and the second are the corresponding target parameters.

        If idx is a list, it returns two tensors as before whose first

        index points to the particular sample and second index points to a value.

        """

        num_features = len(self.training_frame.columns)

        num_targets = len(self.target_frame.columns)



        if type(idx) is list:

            num_samples = len(idx)

            if 'noisy_decay' in self.training_frame.columns:

                # whenever noisy decay is to be included in training data

                trajectory = torch.empty(num_samples,

                                         self.traj_len * (num_features-1) + self.num_times,

                                         dtype=torch.float64)

            else:

                trajectory = torch.empty(num_samples,

                                         self.traj_len * num_features,

                                         dtype=torch.float64)



            target = torch.empty(num_samples,

                                 num_targets,

                                 dtype=torch.float64)

            for i,sample in enumerate(idx):

                # stack trajectories side-by-side (horizontally)

                trajectory[i,:] = torch.cat([ self.training_frame.iloc[sample,feature]

                                              for feature in range(num_features) ])

                target[i,:] = torch.tensor([ self.target_frame.iloc[sample,feature]

                                            for feature in range(num_targets) ])

        elif type(idx) is int:

            if 'noisy_decay' in self.training_frame.columns:

                trajectory = torch.empty(self.traj_len * (num_features-1) + self.num_times,

                                         dtype=torch.float64)

            else:

                trajectory = torch.empty(self.traj_len * num_features,

                                         dtype=torch.float64)



            trajectory = torch.cat([ self.training_frame.iloc[idx,feature]

                                     for feature in range(num_features) ])

            target = torch.empty(num_targets, dtype=torch.float64)

            target = torch.tensor([ self.target_frame.iloc[idx,feature]

                                    for feature in range(num_targets) ],

                                  dtype=torch.double)           

       # print(trajectory)
        return trajectory, target







    def showRegTraj(self, idx, fig_path='.'):

        """

        Plots a given the trajectory input to the neural net against the

        regularization parameter.

        

        Input:

        ------

        idx (int or Tensor of int) - If an int, the index of the trajectory to 

            plot.  If a Tensor of int, a list of indices to plot.

        """

        if type(idx) is int:

            training_traj_names = self.training_frame.columns

            n_plots = len(training_traj_names)

            figure, ax = plt.subplots(nrows=n_plots, ncols=1, sharex=True);



            for i, trajectory in enumerate(training_traj_names):

#                print(f'Trajectory {i+1} of {n_plots} is '+str(trajectory))

                ax[i].semilogx(self.master_frame['reg_params'][idx],

                               self.training_frame[trajectory][idx],

                               linestyle='-', color='black', linewidth=1.0,

                               marker='o', markersize=3.5)

                ax[i].set_ylabel(trajectory)

            figure.savefig(fig_path+'traj_plt.png', dpi=300)





        elif type(idx) is list: # a list of samples is given

            num_samples = len(idx)

            training_traj_names = self.training_frame.columns

            n_plots = len(training_traj_names) # number of subplots in one fig

            for j, sample_idx in enumerate(idx):

                print(f'Starting figure for sample index {sample_idx}: sample {j} of {num_samples}')



                figure, ax = plt.subplots(nrows=n_plots, ncols=1, sharex=True);



                for i, trajectory in enumerate(training_traj_names):

                    print(f'Trajectory {i+1} of {n_plots} is '+str(trajectory))

                    ax[i].semilogx(self.master_frame['reg_params'][sample_idx],

                                   self.training_frame[trajectory][sample_idx],

                                   linestyle='-', color='black', linewidth=1.0,

                                   marker='o', markersize=3.5)

                    ax[i].set_ylabel(trajectory)

                figure.savefig(fig_path+'traj_plt_sample'+str(sample_idx)+'.png',

                               dpi=300)

        return

    









#===============================================================================#





class NonregularizedDecaysDataSet(Dataset):



    def __init__(self, csv_file, target_names, time_bounds, concat=1):

        """

        Input:

        ------

        1. csv_file (string): Path to the csv file containing the data set. This 

            is a saved DataFrame.

        2. target_names (list of strings): The names of the target values the NN

            aims to predict. These are keys in the DataFrame saved to the csv

            file that csv_file points to.

        3. time_bounds (list of 2 floats): Bounds on the times the decay is

            measured at.  The lower bound is time_bounds[0] and the upper bound

            is time_bounds[1].

        4. concat (integer): When an integer greater than 1 is inputted, the time series will

            be duplicated by the input number and concatenated. When the integer is

            equal to 1, no duplication or concatenation will occur.

        """

        self.master_frame = torch.load(csv_file)

        # Targets for the NN to predict

        self.target_frame   = self.master_frame[target_names]

        # number of times at which the decay is sampled 

        self.num_times = len(self.master_frame['noisy_decay'][0])

        # exact times at which the decay is sampled

        self.times = torch.linspace(time_bounds[0],

                                 time_bounds[1],

                                 self.num_times,

                                dtype=torch.float64)

        self.concat = concat

        

    def __len__(self):

        """

        Returns the number of samples in the data set

        """

        return len(self.master_frame)



    def __getitem__(self, idx):

        """

        Returns samples at idx. 

        - If idx is an int, the output is a pair of

        1-d Tensors: the first is the decays obtained from the solution of the 

        NLLS problem with lambda=0 regularization, and the second are the

        corresponding target parameters. 

        - If idx is a list, it returns two Tensors, each with 2 dimensions. The

        first index points to the particular sample and the second index points

        to an intensity at some time for the nonregularized decay.

        """

        # number of variables the NN is trying to predict

        num_targets = len(self.target_frame.columns)

        if self.concat == 1:
            nonregularized_decay = torch.empty(self.num_times, dtype=torch.float64)

            nonregularized_decay = myTrueModel_2param(self.times,

                                                      self.master_frame['c1_target'][idx],

                                                      1.0 - self.master_frame['c1_target'][idx],

                                                      self.master_frame['T21_nlls'][idx],

                                                      self.master_frame['T22_nlls'][idx],

                                                      signalType='biexponential')

        if self.concat > 1:
            nonregularized_decay = torch.empty(self.concat * self.num_times, dtype=torch.float64)

            the_signal = myTrueModel_2param(self.times,

                                            self.master_frame['c1_target'][idx],

                                            1.0 - self.master_frame['c1_target'][idx],

                                            self.master_frame['T21_nlls'][idx],

                                            self.master_frame['T22_nlls'][idx],

                                            signalType='biexponential')

            nonregularized_decay = torch.cat([the_signal for i in range(0, self.concat)])

        target = torch.empty(num_targets, dtype=torch.float64)

        target = torch.tensor([self.target_frame.iloc[idx, feature]

                               for feature in range(num_targets)],

                              dtype=torch.double)

        #print(nonregularized_decay)
        return nonregularized_decay, target





    



#===============================================================================#

    

        







class RegularizedDecaysDataSet(Dataset):



    def __init__(self, csv_file, target_names, time_bounds):

        """

        Input:

        ------

        1. csv_file (string): Path to the csv file containing the data set. This

            is a saved DataFrame.

        2. target_names (list of strings): The names of the target values the NN

            aims to predict. These are keys in the DataFrame saved to the csv

            file that csv_file points to.

        3. time_bounds (list of 2 floats): Bounds on the times the decay is

            measured at.  The lower bound is time_bounds[0] and the upper bound

            is time_bounds[1].

        """

        self.master_frame = torch.load(csv_file)

        # triples (c1_lambda,T21_lambda,T22_lambda) used to define regularized...

        # ...decays in the training data

        self.training_trios_frame = self.master_frame[['c1_traj',

                                                       'T21_traj',

                                                       'T22_traj']]

        self.target_frame   = self.master_frame[target_names]

        # number of regularization parameters and NLLS solution

        self.traj_len  = len(self.master_frame['reg_params'][0]) + 1

        self.num_times = len(self.master_frame['noisy_decay'][0])

        self.times = torch.linspace(time_bounds[0],

                                 time_bounds[1],

                                 self.num_times,

                                dtype=torch.float64)



    def __len__(self):

        """

        Returns the number of samples in the data set

        """

        return len(self.master_frame)



    def __getitem__(self, idx):

        """

        Returns samples at idx.

        - If idx is an int, the output is a pair of 1-d Tensors: the first is

        the regularized decays stacked horizontally and the second is the

        corresponding target parameters.

        - If idx is a list, it returns two Tensors, each with 2 dimensions.

        The first index points to the particular sample and the second index

        points to an intensity at some time for some regularized decay.

        """

        num_targets = len(self.target_frame.columns)

        # number of elements in 1 concatenated regularized decay

        concat_len = self.traj_len * self.num_times

        # the regularized NLLS solutions (c1_ld, T21_ld, T22_ld) for the given sample idx

        regularized_triples = zip(self.training_trios_frame['c1_traj'][idx],

                                  self.training_trios_frame['T21_traj'][idx],

                                  self.training_trios_frame['T22_traj'][idx])

        regularized_decays = torch.empty(concat_len, dtype=torch.float64)

        # gives the nonregularized decay first

        regularized_decays[0:self.num_times] = myTrueModel_2param(self.times,

                                                                  self.master_frame['c1_target'][idx],

                                                                  1.0 - self.master_frame['c1_target'][idx],

                                                                  self.master_frame['T21_nlls'][idx],

                                                                  self.master_frame['T22_nlls'][idx],

                                                                  signalType='biexponential')

        # gives the regularized decays

        regularized_decays[self.num_times:concat_len] = torch.cat([myTrueModel_2param(self.times,

                                                                                      c1,

                                                                                      1.0 - c1,

                                                                                      T21,

                                                                                      T22,

                                                                                      signalType='biexponential')

                                                                   for c1, T21, T22 in regularized_triples])

        target = torch.empty(num_targets, dtype=torch.float64)

        target = torch.tensor([self.target_frame.iloc[idx, feature]

                               for feature in range(num_targets)],

                              dtype=torch.double)

        #print(regularized_decays)
        return regularized_decays, target















#===============================================================================#

class MultipleDecayDataset(Dataset):

    def __init__(self, csv_file, target_names, time_bounds, decay_input=["A", "B", "ND","Const","Rand"]):

        """

        Input:

        ------

        1. csv_file (string): Path to the csv file containing the data set. This

            is a saved DataFrame.

        2. target_names (list of strings): The names of the target values the NN

            aims to predict. These are keys in the DataFrame saved to the csv

            file that csv_file points to.

        3. time_bounds (list of 2 floats): Bounds on the times the decay is

            measured at.  The lower bound is time_bounds[0] and the upper bound

            is time_bounds[1].

        4. decay_input (List of strings): When making an instance of the

        Multiple Decay DataSet, the names of the decay input

        to be included. For example, 'A', 'B', or 'ND'.

        """



        self.master_frame = torch.load(csv_file)

        # triples (c1_lambda,T21_lambda,T22_lambda) used to define regularized...

        # ...decays in the training data

        self.training_trios_frame = self.master_frame[['c1_traj',

                                                       'T21_traj',

                                                       'T22_traj']]


        self.target_frame = self.master_frame[target_names]

        # number of regularization parameters and NLLS solution

        print(self.master_frame['reg_params'][0])
        self.traj_len = len(self.master_frame['reg_params'][0])
        print(self.traj_len)
        self.num_times = len(self.master_frame['noisy_decay'][0])

        self.times = torch.linspace(time_bounds[0],

                                    time_bounds[1],

                                    self.num_times,

                                    dtype=torch.float64)
        self.decay_input = decay_input

    def __len__(self):

        """

        Returns the number of samples in the data set

        """

        return len(self.master_frame)

    def __getitem__(self, idx):

        """

        Returns samples at idx.

        - If idx is an int, the output is a pair of 1-d Tensors: the first is

        the regularized decays stacked horizontally and the second is the

        corresponding target parameters.

        - If idx is a list, it returns two Tensors, each with 2 dimensions.

        The first index points to the particular sample and the second index

        points to an intensity at some time for some regularized decay.

        """

        num_targets = len(self.target_frame.columns)

        # number of elements in 1 concatenated regularized decay

        concat_len = self.traj_len * self.num_times

        decay_input_len = len(self.decay_input) * self.num_times


        list_of_decays = []
        for decay_type in self.decay_input:
            if decay_type == 'A':
                A_tensor = torch.empty(self.num_times, dtype=torch.float64)
                # gives the nonregularized decay
                A_tensor = myTrueModel_2param(self.times,
                                             self.master_frame['c1_target'][idx],
                                             1.0 - self.master_frame['c1_target'][idx],
                                             self.master_frame['T21_nlls'][idx],
                                             self.master_frame['T22_nlls'][idx],
                                             signalType='biexponential')

                list_of_decays.append(A_tensor)

            if decay_type == 'B':
                for ld in range(self.traj_len):
                    B_tensor = torch.empty(self.num_times, dtype=torch.float64)
                    # gives the regularized decay
                    #print('B1', B_tensor)
                    B_tensor = myTrueModel_2param(self.times,
                                              self.training_trios_frame['c1_traj'][idx][ld],
                                            (1.0 - self.training_trios_frame['c1_traj'][idx][ld]),
                                              self.training_trios_frame['T21_traj'][idx][ld],
                                              self.training_trios_frame['T22_traj'][idx][ld],
                                              signalType='biexponential')
                    list_of_decays.append(B_tensor)
                    #print('B2', B_tensor)
                    #print(type(B_tensor))


            if decay_type == 'ND':
                ND_tensor = torch.empty(self.num_times, dtype=torch.float64)
                # gives the noisy decay
                #print('ND1', ND_tensor)
                ND_tensor = self.master_frame['noisy_decay'][idx]
                #print('ND2', ND_tensor)
                #print(type(ND_tensor))

                list_of_decays.append(ND_tensor)
                
            if decay_type == 'Const':
                Const_tensor = torch.empty(self.num_times, dtype=torch.float64)
                # gives the noisy decay
                #print(Const_tensor)
                temporary = []
                for i in range(64):
                    temporary.append(1)
                Const_tensor = torch.FloatTensor(temporary)
                #print(Const_tensor)
                
                list_of_decays.append(Const_tensor)
                
            if decay_type == 'Rand':
                Rand_tensor = torch.empty(self.num_times, dtype=torch.float64)
                # gives the noisy decay
                #print(Rand_tensor)
                temporary = []
                for i in range(64):
                    temporary.append(random.random())
                Rand_tensor = torch.FloatTensor(temporary)
                #print(Rand_tensor)
                
                list_of_decays.append(Rand_tensor)

        multiple_decays = torch.cat(list_of_decays)

        target = torch.empty(num_targets, dtype=torch.float64)

        target = torch.tensor([self.target_frame.iloc[idx, feature]

                               for feature in range(num_targets)],

                              dtype=torch.double)

        # print(multiple_decays)
        return multiple_decays, target


#################################################################################


def initDataSet(nn_input_type,

                csv_path,

                target_names,

                time_bounds,

                *,

                trajectories=[],

                concat=1,

                noisy_concat=1,

                decay_input=[]):

    """

    INITDATASET initializes a dataset necessary for training and or testing

    a neural net.



    INPUT:

    ------

    1. nn_input_type (String) - Describes the input to the NN. (Currently) one of 

        'NoisyDecays', 'NonregularizedDecays', 'RegularizedDecays', or 

        'RegularizationTrajectories'.

    2. csv_path (String) - path to the data stored as a csv

    3. target_names (List of strings) - The names of the target values the NN

        aims to predict. These are keys in the DataFrame saved to the csv

        file to which csv_path points.

    4. time_bounds (Array of floats) - The time of the first measurement of

        the signal and the time of the last measurement.

    5. trajectories (List of strings) - When making an instance of the 

        RegularizationTrajectoryDataSet, the names of the particular trajectories

        to be included. For example, 'c1_traj', 'T21_traj', 'T22_traj', 

        or 'noisy_decay'.

    6. concat (integer): concat is an argument only used with the NonRegularized Decay dataset.

        When an integer greater than 1 is inputted, the time series will be duplicated by the

        input number and concatenated. When the integer is equal to 1, no duplication or concatenation

        will occur.
    7. noisy concat (integer): concat is an argument only used with the Noisy Decay dataset.

        When an integer greater than 1 is inputted, the time series will be duplicated by the

        input number and concatenated. When the integer is equal to 1, no duplication or concatenation

        will occur.

    8. decay_input (List of strings): When making an instance of the

        Multiple Decay DataSet, the names of the decay input

        to be included. For example, 'A', 'B', or 'ND'.




    OUTPUT:

    -------

    1. dataset (instance of torch.utils.data.Dataset) - instance of the desired

        data set.

    """

        

    if nn_input_type is 'NoisyDecays':

        dataset = NoisyDecaysDataSet(csv_path,

                                     target_names,

                                     time_bounds,

                                     noisy_concat)

    elif nn_input_type is 'Noisy_RegularizedDecays':
        dataset = Noisy_RegularizedDecaysDataSet(csv_path,

                                           target_names,

                                           time_bounds)



    elif nn_input_type is 'NonregularizedDecays':

        dataset = NonregularizedDecaysDataSet(csv_path,

                                              target_names,

                                              time_bounds,

                                              concat)



    elif nn_input_type is 'RegularizedDecays':

        dataset = RegularizedDecaysDataSet(csv_path,

                                           target_names,

                                           time_bounds)



    elif nn_input_type is 'RegularizationTrajectories':

        dataset = RegularizationTrajectoryDataSet(csv_path,

                                                  trajectories,

                                                  target_names,

                                                  time_bounds)
    elif nn_input_type is 'MultipleDecays':

        dataset = MultipleDecayDataset(csv_path,

                                           target_names,

                                           time_bounds,

                                           decay_input)

    return dataset













#===============================================================================#













def methodLabel(nn_input_type, num_copies=1, num_noisy_copies=1, decay_input = ['ND','B'], regularization_parameters=[]):

    """

    METHODLABEL gives a string and abbreviation describing the neural net input

    type.  This is useful for plotting and printing output.



    INPUT:

    ------

    1. nn_input_type (String) - Describes the input to the NN. (Currently) one of 

        'NoisyDecays', 'NonregularizedDecays', 'Concatenated Nonregularized Decays',

         'RegularizedDecays', or 'RegularizationTrajectories'.

    2. testing_dataset (instance of torch.utils.data.Dataset) - instance of the

        NN's testing data set.

    3. num_copies (Integer) - If the input type is a Nonregularized Decays, num_copies

        indicates how many times the input to the NN is duplicated and concatenated

    4. num_noisy_copies (Integer) - If the input type is a Noisy Decays, num_noisy_copies

        indicates how many times the input to the NN is duplicated and concatenated

    5. decay_input (List of strings): When making an instance of the

        Multiple Decay DataSet, the names of the decay input

        to be included. For example, 'A', 'B', or 'ND'.




    OUTPUT:

    -------

    1. label (String) - a descriptive string describing the NN input.

    2. abr (String) - abbreviation for the NN input.

    """

        

    if nn_input_type is 'NoisyDecays' and num_noisy_copies == 1:

        label, abr = 'Noisy Decays', 'ND'


    elif nn_input_type is 'NoisyDecays' and num_noisy_copies >= 2:

        label, abr = 'Concatenated Noisy Decays', 'ND'*(num_noisy_copies)


    elif nn_input_type is 'Noisy_RegularizedDecays':

        label, abr = 'Noisy and Regularzied Decays', 'NDB'


    elif nn_input_type is 'NonregularizedDecays' and num_copies == 1:

        label, abr = 'Nonregularized Decays', 'A'


    elif nn_input_type is 'NonregularizedDecays' and num_copies >= 2:

        label, abr = 'Concatenated Nonregularized Decays', 'A'*(num_copies)


    elif nn_input_type is 'RegularizedDecays':

        label, abr = 'Regularized Decays', 'AB'


    elif nn_input_type is 'RegularizationTrajectories':

        label, abr = 'Regularization Trajectories', 'RT'


    elif nn_input_type is 'MultipleDecays':
        if 'B' in decay_input:
            label_input = decay_input
            label_input.remove('B')
            B_params = len(regularization_parameters)
            for b in range(0,B_params):
                B_string = 'B'+str(b+1)
                label_input.append(B_string)
            label, abr = 'Multiple Decays', 'MD ('+ " ".join(str(x) for x in label_input)+")"
        else:
            label, abr = 'Multiple Decays', 'MD ('+ " ".join(str(x) for x in decay_input)+")"

    return label, abr













#===============================================================================#













def calcInDim(nn_input_type, testing_dataset, concat=1, noisy_concat=1, decay_input = []):

    """

    Calculates the input dimension of the neural net (NN).

    

    INPUT:

    ------

    1. nn_input_type (String) - Describes the input to the NN. (Currently) one of 

        'NoisyDecays', 'NonregularizedDecays', 'RegularizedDecays', or 

        'RegularizationTrajectories'.

    2. testing_dataset (instance of torch.utils.data.Dataset) - instance of the

        NN's testing data set.

    3. concat (integer): When an integer greater than 1 is inputted, the time series will be duplicated by the

        input number and concatenated. When the integer is equal to 1, no duplication or concatenation

        will occur.

     5. decay_input (List of strings): When making an instance of the

        Multiple Decay DataSet, the names of the decay input

        to be included. For example, 'A', 'B', or 'ND'.



    OUTPUT:

    -------

    1. in_dim (int) - width of the input layer to the NN.

    """



    if (nn_input_type is 'NoisyDecays' and noisy_concat==1) or (nn_input_type is 'NonregularizedDecays' and concat==1):

        # The input is a decay measured at fixed times.



        # number of times the decay is measured



        in_dim = testing_dataset.num_times

    elif nn_input_type is 'Noisy_RegularizedDecays':

        # The input is a vector of concatenated, regularized decays.



        # number of regularization parameters

        num_reg_params = testing_dataset.traj_len

        # number of times the decay is measured

        num_times = testing_dataset.num_times

        in_dim = num_reg_params*num_times




    elif concat > 1:

        in_dim = testing_dataset.num_times*concat

    elif noisy_concat > 1:

        in_dim = testing_dataset.num_times*noisy_concat



    elif nn_input_type is 'RegularizedDecays':

        # The input is a vector of concatenated, regularized decays.



        # number of regularization parameters

        num_reg_params = testing_dataset.traj_len

        # number of times the decay is measured

        num_times = testing_dataset.num_times

        in_dim = num_reg_params*num_times



    elif nn_input_type is 'RegularizationTrajectories':

        # Let (c1_l, T21_l, T22_l) be the solution to the regularized NLLS...

        # ...problem with parameter l. Then (c1_l)_{l in L} is the...

        # ...'regularization trajectory' of c1, (T21_l)_{l in L} is the...

        # ...regularization trajectory of T21, and so on.  The input is...

        # ... concatenated regularization trajecories, and possibly the

        # ... noisy decay.



        # number of regularization parameters

        num_reg_params = testing_dataset.traj_len

        

        # number of times the decay is measured

        num_times = testing_dataset.num_times

        # number of regularization trajectories

        particular_trajectories = testing_dataset.training_frame.columns

        num_trajectories = len(particular_trajectories)

        if 'noisy_decay' in particular_trajectories: num_trajectories -= 1

        in_dim = num_reg_params*num_trajectories

        if 'noisy_decay' in particular_trajectories: in_dim += num_times

    elif nn_input_type is 'MultipleDecays':

        # The input is a vector of concatenated decays.

        # number of times the decay is measured

        num_times = testing_dataset.num_times

        B_count = decay_input.count('B')

        if 'B' in decay_input:
            in_dim = (len(decay_input) - (B_count))*num_times + B_count*testing_dataset.traj_len*num_times
        else:
            in_dim = len(decay_input)*num_times



    return in_dim

        





















