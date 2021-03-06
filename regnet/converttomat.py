import torch 
import os 
import numpy as np 
from scipy.io import savemat
print(os.getcwd())
files = os.popen("ls pthdata").read().split("\n")
files = files[:-1]
for file in files:
    model = torch.load(os.getcwd() + '/pthdata/' + file, map_location=torch.device('cpu'))
    # savemat(os.getcwd()+'/matdata/'+file[:-4]+'.mat', {"w1": model['inputMap.weight'].numpy(), 
    #                                                    "w2": model['hidden1.weight'].numpy(), 
    #                                                    "w3": model['hidden2.weight'].numpy(), 
    #                                                    "w4": model['hidden3.weight'].numpy(),
    #                                                    "w5": model['outputMap.weight'].numpy()})
    savemat(os.getcwd()+'/matdata/'+file[:-4]+'.mat', model)
