import torch
import torch.nn as nn
import torch.nn.functional as F




#--------------------------#
## Define NN architecture ##
#--------------------------#
# A) Vanilla Parameter Estimation
#--------------------------------
#    num_times is the number of signal acquisition times
#    input: tensor of size P['batch_size'] x num_times
#    output: tensor of size P['batch_size'] x num_param
#            For biexponential, num_param=3:
#              initial fraction c1, time constants T21,T22

class fullyConnectedNN(nn.Module):

    def __init__(self, in_dim, h1_dim, h2_dim, h3_dim, h4_dim, out_dim):
        super(fullyConnectedNN, self).__init__()
        self.inputMap  = nn.Linear(in_dim, h1_dim)
        self.hidden1   = nn.Linear(h1_dim, h2_dim)
        self.hidden2   = nn.Linear(h2_dim, h3_dim)
        self.hidden3   = nn.Linear(h3_dim, h4_dim)
        self.outputMap = nn.Linear(h4_dim, out_dim)

    def forward(self,x):
        x = F.relu(self.inputMap(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.outputMap(x)
        return x



    
#===============================================================================#





# B) Parameter Estimation with L^2 Regularization Trajectory
#-----------------------------------------------------------
#    input: tensor of size P['batch_size'] x num_param x num_ld
#    output: tensor of size P['batch_size'] x num_param
#            For biexponential, num_param=3:
#              initial fraction c1, time constants T21 and T22

class regularizationTrajNN(nn.Module):

    def __init__(self, in_dim, h1_dim, h2_dim, h3_dim, h4_dim, out_dim):
        super(regularizationTrajNN, self).__init__()
        self.inputMap  = nn.Linear(in_dim, h1_dim)
        self.hidden1   = nn.Linear(h1_dim, h2_dim)
        self.hidden2   = nn.Linear(h2_dim, h3_dim)
        self.hidden3   = nn.Linear(h3_dim, h4_dim)
        self.outputMap = nn.Linear(h4_dim, out_dim)

    def forward(self,x):
        x = F.relu(self.inputMap(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.outputMap(x)
        return x







#===============================================================================#



# C) Binary Model Labeling Problem: Probability Output with softmax
#------------------------------------------------------------------
#    input: tensor of size P['batch_size'] x num_param x num_ld
#    output: tensor of size P['batch_size'] x 2
#            For biexponential, num_param=3:
#              initial fraction c1, time constants T21 and T22

class BinaryModelLabelNN_Probability(nn.Module):

    def __init__(self, in_dim, h1_dim, h2_dim, h3_dim, h4_dim, out_dim):
        super(BinaryModelLabelNN_Probability, self).__init__()
        self.inputMap = nn.Linear(in_dim, h1_dim)
        self.hidden1 = nn.Linear(h1_dim, h2_dim)
        self.hidden2 = nn.Linear(h2_dim, h3_dim)
        self.hidden3 = nn.Linear(h3_dim, h4_dim)
        self.outputMap = nn.Linear(h4_dim, out_dim)

    def forward(self,x):
        x = F.relu(self.inputMap(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.softmax(self.outputMap(x), dim=1) # returns probability distribution
        return x
    





#===============================================================================#






# D) Binary Model Labeling Problem: Score output
#-----------------------------------------------
#    input: tensor of size P['batch_size'] x (num_param * num_ld) (for example)
#    output: tensor of size P['batch_size'] x 2
#            For biexponential, num_param=3:
#              initial fraction c1, time constants T21 and T22

class BinaryModelLabelNN_Score(nn.Module):

    def __init__(self, in_dim, h1_dim, h2_dim, h3_dim, h4_dim, out_dim):
        super(BinaryModelLabelNN_Score, self).__init__()
        self.inputMap = nn.Linear(in_dim, h1_dim)
        self.hidden1 = nn.Linear(h1_dim, h2_dim)
        self.hidden2 = nn.Linear(h2_dim, h3_dim)
        self.hidden3 = nn.Linear(h3_dim, h4_dim)
        self.outputMap = nn.Linear(h4_dim, out_dim)

    def forward(self,x):
        x = F.relu(self.inputMap(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.outputMap(x)
        return x





# ===============================================================================#







# E) Multilayer Perceptron (HL1)
# --------------------------------
#    num_times is the number of signal acquisition times
#    input: tensor of size P['batch_size'] x num_times
#    output: tensor of size P['batch_size'] x num_param
#            For biexponential, num_param=3:
#              initial fraction c1, time constants T21,T22

class multilayerPerceptronHL1(nn.Module):

    def __init__(self, in_dim, h1_dim, out_dim):
        super(multilayerPerceptronHL1, self).__init__()
        self.InputtoHidden1 = nn.Linear(in_dim, h1_dim)
        self.Hidden1toOutput = nn.Linear(h1_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.InputtoHidden1(x))
        x = self.Hidden1toOutput(x)
        return x

    # ===============================================================================#







# F) Multilayer Perceptron (HL2)
# --------------------------------
#    num_times is the number of signal acquisition times
#    input: tensor of size P['batch_size'] x num_times
#    output: tensor of size P['batch_size'] x num_param
#            For biexponential, num_param=3:
#              initial fraction c1, time constants T21,T22

class multilayerPerceptronHL2(nn.Module):

    def __init__(self, in_dim, h1_dim, h2_dim, out_dim):
        super(multilayerPerceptronHL2, self).__init__()
        self.InputtoHidden1 = nn.Linear(in_dim, h1_dim)
        self.Hidden1toHidden2 = nn.Linear(h1_dim, h2_dim)
        self.Hidden2toOutput = nn.Linear(h2_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.InputtoHidden1(x))
        x = F.relu(self.Hidden1toHidden2(x))
        x = self.Hidden2toOutput(x)
        return x

    # ===============================================================================#







# G) Multilayer Perceptron (HL3)
# --------------------------------
#    num_times is the number of signal acquisition times
#    input: tensor of size P['batch_size'] x num_times
#    output: tensor of size P['batch_size'] x num_param
#            For biexponential, num_param=3:
#              initial fraction c1, time constants T21,T22

class multilayerPerceptronHL3(nn.Module):

        def __init__(self, in_dim, h1_dim, h2_dim, h3_dim, out_dim):
            super(multilayerPerceptronHL3, self).__init__()
            self.InputtoHidden1 = nn.Linear(in_dim, h1_dim)
            self.Hidden1toHidden2 = nn.Linear(h1_dim, h2_dim)
            self.Hidden2toHidden3 = nn.Linear(h2_dim, h3_dim)
            self.Hidden3toOutput = nn.Linear(h3_dim, out_dim)

        def forward(self, x):
            x = F.relu(self.InputtoHidden1(x))
            x = F.relu(self.Hidden1toHidden2(x))
            x = F.relu(self.Hidden2toHidden3(x))
            x = self.Hidden3toOutput(x)
            return x

    # ===============================================================================#


# H) Expanded Liu Network
#-----------------------------------------------------------
#    input: tensor of size P['batch_size'] x num_param x num_ld
#    output: tensor of size P['batch_size'] x num_param
#            For biexponential, num_param=3:
#              initial fraction c1, time constants T21 and T22

class ExpandedLengthNN(nn.Module):

    def __init__(self, in_dim, h1_dim, h2_dim, h3_dim, h4_dim, h5_dim, out_dim):
        super(ExpandedLengthNN, self).__init__()
        self.inputMap  = nn.Linear(in_dim, h1_dim)
        self.hidden1   = nn.Linear(h1_dim, h2_dim)
        self.hidden2   = nn.Linear(h2_dim, h3_dim)
        self.hidden3   = nn.Linear(h3_dim, h4_dim)
        self.hidden4 = nn.Linear(h4_dim, h5_dim)
        self.outputMap = nn.Linear(h5_dim, out_dim)

    def forward(self,x):
        x = F.relu(self.inputMap(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = self.outputMap(x)
        return x
