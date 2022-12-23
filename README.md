# descrambling-NN

This is the code repository for the paper "Emergence of the SVD as an interpretable factorization in deep learning for inverse problems" by Sule, Czaja, Spencer. To explore our code and figures, first download the data files into the requisite `/regnet/` and `/2_Layer_DeerNet/` folders from [here](https://umd.box.com/s/zul3iaq9x6u332nydsc6lo19uvote8ry). You will need MATLAB 2022a with the DeepLearning and Symbolic Computation toolboxes downloaded. 
Additionally, if you wish to train a neural network, we recommend using Pytorch with preferable an NVIDIA GPU. 

1. **DEERNet** We have trained shallow DEERNets with both ReLU and Sigmoid intermediate activations. You can access these networks, their tranining sets, and descrambler matrices by opening the `mat` files starting with `workspace...` downloaded from the above link to our data. 

2. **ILR networks** We recommend first setting up an conda environment using the `py38_NN.yml` file in the `/regnet/` regnet directory. Trained ILR networks are available in the `/regnet/matdata/` directory and datasets are available in the `/regnet/_results/_datasets/` directory. To generate descrambling analyses of different layers you should run the `regnetdata.m` file in the `/regnet/` folder. The code used to generate the biexponential fits in Figure 4 is `biexponential_nlls.m`. 

