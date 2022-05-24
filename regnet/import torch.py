import torch 
import os 
import numpy as np 

files = os.popen("ls pthdata").read()
for file in files:
    model = torch.load(os.getcwd() + '/pthdata/' + file, map_location=torch.device('cpu'))
    print(file[:-4])