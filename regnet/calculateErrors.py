import torch
import numpy as np

class errorMethods:
    
    def log_mean_percent_error(y_predicted, y_target):
        """
        Computes the mean percent error for each column in y_predicted. If only
        two columns are present in y_predicted, it assumes T21 and T22 are being
        predicted. If three columns are present, it assumes c1 is the first column.

        Input:
        ------
        1. y_predicted (Tensor of size (num_batch, *)): Values predicted by NN
        2. y_target (Tensor of size (num_batch, *)): Target values

        Output:
        -------
        1. mean_percent_error (array): average percent error of each column
           of y_predicted.
        """
        perc_error_denom = y_target.detach().clone()
        if len(y_predicted[0,:]) == 3:
            # calculate relative error of c2 since it's bounded away from 0
            perc_error_denom[:,0] = 1.0 - perc_error_denom[:,0]
        percent_error = 100.0*torch.abs(torch.log(y_predicted) - torch.log(y_target))/perc_error_denom #I just added log
        return torch.mean(percent_error, dim=0) # average over samples
    
    
    def mean_percent_error(y_predicted, y_target):
        """
        Computes the mean percent error for each column in y_predicted. If only
        two columns are present in y_predicted, it assumes T21 and T22 are being
        predicted. If three columns are present, it assumes c1 is the first column.

        Input:
        ------
        1. y_predicted (Tensor of size (num_batch, *)): Values predicted by NN
        2. y_target (Tensor of size (num_batch, *)): Target values

        Output:
        -------
        1. mean_percent_error (array): average percent error of each column
           of y_predicted.
        """
        perc_error_denom = y_target.detach().clone()
        if len(y_predicted[0,:]) == 3:
           # calculate relative error of c2 since it's bounded away from 0
           perc_error_denom[:,0] = 1.0 - perc_error_denom[:,0]
        percent_error = 100.0*torch.abs(y_predicted - y_target)/perc_error_denom
        return torch.mean(percent_error, dim=0) # average over samples
    
    def log_mean_square_error(y_predicted, y_target):
        """
        Computes the mean square error for each column in y_predicted. If only
        two columns are present in y_predicted, it assumes T21 and T22 are being
        predicted. If three columns are present, it assumes c1 is the first column.

        Input:
        ------
        1. y_predicted (Tensor of size (num_batch, *)): Values predicted by NN
        2. y_target (Tensor of size (num_batch, *)): Target values

        Output:
        -------
        1. mean_square_error (array): average percent error of each column
           of y_predicted.
        """
        MSE = torch.square(torch.log(y_target) - torch.log(y_predicted))
        return torch.mean(MSE, dim=0) # average over samples
    
    def mean_square_error(y_predicted, y_target):
        """
        Computes the mean percent error for each column in y_predicted. If only
        two columns are present in y_predicted, it assumes T21 and T22 are being
        predicted. If three columns are present, it assumes c1 is the first column.

        Input:
        ------
        1. y_predicted (Tensor of size (num_batch, *)): Values predicted by NN
        2. y_target (Tensor of size (num_batch, *)): Target values

        Output:
        -------
        1. mean_percent_error (array): average percent error of each column
           of y_predicted.
        """
        MSE = torch.square(y_target - y_predicted)
        return torch.mean(MSE, dim=0) # average over samples
    
    def mask_outliers(data, tol=1.5):
        """
        MASK_OUTLIERS masks the outliers in the Tensor error according to a 
          boxplot. Let Q1, Q2 (i.e., median), and Q3 be the first, second, and
          third quartiles of the data. Let IQR := Q3-Q1 be the interquartile
          range. A member of data is considered an outlier if it is less than 
          Q1 - tol*IQR or greater than Q3 + tol*IQR.


        INPUT:
        ------
        - data (Tensor) - Data we wish to discard outliers from
        - tol (float) - Tolerance for an outlier.  

        OUTPUT:
        -------
        - mask (Boolean Tensor) -  Tensor with the same shape as data. An element 
          in mask is True if the corresponding element in data is good and False 
          if the corresponding element in data is an outlier.
        """
        # First quartile. Member of data that 25% of the data is less than
        Q1 = np.quantile(data, 0.25, interpolation='lower')
        # Second quartile (i.e., median)
        Q2 = np.quantile(data, 0.5, interpolation='lower')
        # Third quartile. Member of data that 75% of the data is less than 
        Q3 = np.quantile(data, 0.75, interpolation='lower')
        # Interquartile range
        IQR = Q3 - Q1
        
        # returns a Boolean Tensor of the same size as perc_err.
        # Elements that are less than tol*stdev_perc_err are marked as True.

        mask = (data >= (Q1 - tol*IQR)) & (data <= (Q3 + tol*IQR))

        # Return mask of good values
        return mask
        
        


    
    def sample_averaged_abs_error_stdev(true, approx):
        abs_error = torch.abs(true - approx)
        return (torch.mean(abs_error, dim=0),
                torch.std(abs_error, dim=0))

    def sample_averaged_error_stdev(true, approx, error):
        if error == 'PE':
            abs_error = torch.abs(true - approx)
            error = 100.0*abs_error/true
        if error == 'SE':
            abs_error = torch.abs(true - approx)
            error = abs_error**2
        if error == 'LPE':
            abs_error = torch.abs(torch.log(true) - torch.log(approx))
            error = 100.0*abs_error/true
        if error == 'LSE':
            abs_error = torch.abs(torch.log(true) - torch.log(approx))
            error = abs_error**2
        return (torch.mean(error, dim=0),
                torch.std(error, dim=0))


    def sample_averaged_perc_error_stdev(true, approx):
        abs_error = torch.abs(true - approx)
        perc_error = 100.0*abs_error/true
        #print('THIS IS THE PERCENT ERROR STDEV', torch.std(perc_error, dim=0))
        return (torch.mean(perc_error, dim=0),
                torch.std(perc_error, dim=0))

    def sample_averaged_squa_error_stdev(true, approx):
        abs_error = torch.abs(true - approx)
        perc_error = abs_error**2
        #print('THIS IS THE SQUARE ERROR STDEV', torch.std(perc_error, dim=0))
        return (torch.mean(perc_error, dim=0),
                torch.std(perc_error, dim=0))
    
    def log_sample_averaged_perc_error_stdev(true, approx):
        abs_error = torch.abs(torch.log(true) - torch.log(approx))
        perc_error = 100.0*abs_error/true
        return (torch.mean(perc_error, dim=0),
                torch.std(perc_error, dim=0))

    def log_sample_averaged_squa_error_stdev(true, approx):
        abs_error = torch.abs(torch.log(true) - torch.log(approx))
        perc_error = abs_error**2
        return (torch.mean(perc_error, dim=0),
                torch.std(perc_error, dim=0))


    def prediction_averages_stdevs(self, true, approx, error):
        # true and approx are 1 dimensional Tensors.
        # The k-th entry of approx is an approximation for the k-th entry of true.

        # Sequence of unique targets, ordered increasingly
        list_of_targets = torch.unique(true, sorted=True)
        num_of_targets  = len(list_of_targets)

        # Initialize containers. Using NaNs makes missing pairs visible in...
        # ...heat maps.
        prediction_means             = np.full(num_of_targets, np.nan, np.float64)
        prediction_stdevs            = np.full(num_of_targets, np.nan, np.float64)
        prediction_error_means       = np.full(num_of_targets, np.nan, np.float64)
        prediction_error_stdevs      = np.full(num_of_targets, np.nan, np.float64)
        twoprediction_error_means  = np.full(num_of_targets, np.nan, np.float64)
        twoprediction_error_stdevs = np.full(num_of_targets, np.nan, np.float64)


        for (i, target) in enumerate(list_of_targets):
            # For this target, its locations amongst the samples
            target_locs = torch.where(true == target)[0]

            # samples corresponding to this target
            these_targets  = true[target_locs]
            these_approxes = approx[target_locs]

            # average prediction and standard deviation
            prediction_means[i] = torch.mean(these_approxes)
            prediction_stdevs[i] = torch.std(these_approxes)

            # average error and standard deviation of error
            prediction_error_means[i], prediction_error_stdevs[i] = self.sample_averaged_abs_error_stdev(these_targets,
                                                                                                         these_approxes)

            # average percent error and standard deviation of percent error
            if error == 'PE':
                twoprediction_error_means[i], twoprediction_error_stdevs[i] =self.sample_averaged_perc_error_stdev(these_targets, these_approxes)
            if error == 'SE':
                twoprediction_error_means[i], twoprediction_error_stdevs[i] =self.sample_averaged_squa_error_stdev(these_targets, these_approxes)
            if error == 'LPE':
                twoprediction_error_means[i], twoprediction_error_stdevs[i] =self.log_sample_averaged_perc_error_stdev(these_targets, these_approxes)
            if error == 'LSE':
                twoprediction_error_means[i], twoprediction_error_stdevs[i] =self.log_sample_averaged_squa_error_stdev(these_targets, these_approxes)
                
        return (list_of_targets, prediction_means, prediction_stdevs,
                prediction_error_means, prediction_error_stdevs,
                twoprediction_error_means, twoprediction_error_stdevs)
    
    def PE_prediction_averages_stdevs(self, true, approx):
        # true and approx are 1 dimensional Tensors.
        # The k-th entry of approx is an approximation for the k-th entry of true.

        # Sequence of unique targets, ordered increasingly
        list_of_targets = torch.unique(true, sorted=True)
        num_of_targets  = len(list_of_targets)

        # Initialize containers. Using NaNs makes missing pairs visible in...
        # ...heat maps.
        prediction_means             = np.full(num_of_targets, np.nan, np.float64)
        prediction_stdevs            = np.full(num_of_targets, np.nan, np.float64)
        prediction_error_means       = np.full(num_of_targets, np.nan, np.float64)
        prediction_error_stdevs      = np.full(num_of_targets, np.nan, np.float64)
        prediction_perc_error_means  = np.full(num_of_targets, np.nan, np.float64)
        prediction_perc_error_stdevs = np.full(num_of_targets, np.nan, np.float64)


        for (i, target) in enumerate(list_of_targets):
            # For this target, its locations amongst the samples
            target_locs = torch.where(true == target)[0]

            # samples corresponding to this target
            these_targets  = true[target_locs]
            these_approxes = approx[target_locs]

            # average prediction and standard deviation
            prediction_means[i] = torch.mean(these_approxes)
            prediction_stdevs[i] = torch.std(these_approxes)

            # average error and standard deviation of error
            prediction_error_means[i], prediction_error_stdevs[i] = self.sample_averaged_abs_error_stdev(these_targets,
                                                                                                         these_approxes)

            # average percent error and standard deviation of percent error
            prediction_perc_error_means[i], prediction_perc_error_stdevs[i] =self.sample_averaged_perc_error_stdev(these_targets,
                                                                                                                   these_approxes)
        return (list_of_targets, prediction_means, prediction_stdevs,
                prediction_error_means, prediction_error_stdevs,
                prediction_perc_error_means, prediction_perc_error_stdevs)
    
    def SE_prediction_averages_stdevs(self, true, approx):
        # true and approx are 1 dimensional Tensors.
        # The k-th entry of approx is an approximation for the k-th entry of true.

        # Sequence of unique targets, ordered increasingly
        list_of_targets = torch.unique(true, sorted=True)
        num_of_targets  = len(list_of_targets)

        # Initialize containers. Using NaNs makes missing pairs visible in...
        # ...heat maps.
        prediction_means             = np.full(num_of_targets, np.nan, np.float64)
        prediction_stdevs            = np.full(num_of_targets, np.nan, np.float64)
        prediction_error_means       = np.full(num_of_targets, np.nan, np.float64)
        prediction_error_stdevs      = np.full(num_of_targets, np.nan, np.float64)
        prediction_squa_error_means  = np.full(num_of_targets, np.nan, np.float64)
        prediction_squa_error_stdevs = np.full(num_of_targets, np.nan, np.float64)

        for (i, target) in enumerate(list_of_targets):
            # For this target, its locations amongst the samples
            target_locs = torch.where(true == target)[0]

            # samples corresponding to this target
            these_targets  = true[target_locs]
            these_approxes = approx[target_locs]

            # average prediction and standard deviation
            prediction_means[i] = torch.mean(these_approxes)
            prediction_stdevs[i] = torch.std(these_approxes)

            # average error and standard deviation of error
            prediction_error_means[i], prediction_error_stdevs[i] = self.sample_averaged_abs_error_stdev(these_targets,
                                                                                                         these_approxes)

            # average square error and standard deviation of square error
            prediction_squa_error_means[i], prediction_squa_error_stdevs[i] =self.sample_averaged_squa_error_stdev(these_targets,
                                                                                                                   these_approxes)
            
        return (list_of_targets, prediction_means, prediction_stdevs,
                prediction_error_means, prediction_error_stdevs,
                prediction_squa_error_means, prediction_squa_error_stdevs)
    
    def LPE_prediction_averages_stdevs(self, true, approx):
        # true and approx are 1 dimensional Tensors.
        # The k-th entry of approx is an approximation for the k-th entry of true.

        # Sequence of unique targets, ordered increasingly
        list_of_targets = torch.unique(true, sorted=True)
        num_of_targets  = len(list_of_targets)

        # Initialize containers. Using NaNs makes missing pairs visible in...
        # ...heat maps.
        prediction_means             = np.full(num_of_targets, np.nan, np.float64)
        prediction_stdevs            = np.full(num_of_targets, np.nan, np.float64)
        prediction_error_means       = np.full(num_of_targets, np.nan, np.float64)
        prediction_error_stdevs      = np.full(num_of_targets, np.nan, np.float64)
        prediction_perc_error_means  = np.full(num_of_targets, np.nan, np.float64)
        prediction_perc_error_stdevs = np.full(num_of_targets, np.nan, np.float64)


        for (i, target) in enumerate(list_of_targets):
            # For this target, its locations amongst the samples
            target_locs = torch.where(true == target)[0]

            # samples corresponding to this target
            these_targets  = true[target_locs]
            these_approxes = approx[target_locs]

            # average prediction and standard deviation
            prediction_means[i] = torch.mean(these_approxes)
            prediction_stdevs[i] = torch.std(these_approxes)

            # average error and standard deviation of error
            prediction_error_means[i], prediction_error_stdevs[i] = self.sample_averaged_abs_error_stdev(these_targets,
                                                                                                         these_approxes)

            # average percent error and standard deviation of percent error
            prediction_perc_error_means[i], prediction_perc_error_stdevs[i] =self.log_sample_averaged_perc_error_stdev(these_targets,
                                                                                                                   these_approxes)

            
        return (list_of_targets, prediction_means, prediction_stdevs,
                prediction_error_means, prediction_error_stdevs,
                prediction_perc_error_means, prediction_perc_error_stdevs)
    
    def LSE_prediction_averages_stdevs(self, true, approx):
        # true and approx are 1 dimensional Tensors.
        # The k-th entry of approx is an approximation for the k-th entry of true.

        # Sequence of unique targets, ordered increasingly
        list_of_targets = torch.unique(true, sorted=True)
        num_of_targets  = len(list_of_targets)

        # Initialize containers. Using NaNs makes missing pairs visible in...
        # ...heat maps.
        prediction_means             = np.full(num_of_targets, np.nan, np.float64)
        prediction_stdevs            = np.full(num_of_targets, np.nan, np.float64)
        prediction_error_means       = np.full(num_of_targets, np.nan, np.float64)
        prediction_error_stdevs      = np.full(num_of_targets, np.nan, np.float64)
        prediction_squa_error_means  = np.full(num_of_targets, np.nan, np.float64)
        prediction_squa_error_stdevs = np.full(num_of_targets, np.nan, np.float64)

        for (i, target) in enumerate(list_of_targets):
            # For this target, its locations amongst the samples
            target_locs = torch.where(true == target)[0]

            # samples corresponding to this target
            these_targets  = true[target_locs]
            these_approxes = approx[target_locs]

            # average prediction and standard deviation
            prediction_means[i] = torch.mean(these_approxes)
            prediction_stdevs[i] = torch.std(these_approxes)

            # average error and standard deviation of error
            prediction_error_means[i], prediction_error_stdevs[i] = self.sample_averaged_abs_error_stdev(these_targets,
                                                                                                         these_approxes)

            # average log square error and standard deviation of log square error
            prediction_squa_error_means[i], prediction_squa_error_stdevs[i] =self.log_sample_averaged_squa_error_stdev(these_targets,
                                                                                                                   these_approxes)
            
        return (list_of_targets, prediction_means, prediction_stdevs,
                prediction_error_means, prediction_error_stdevs,
                prediction_squa_error_means, prediction_squa_error_stdevs)

    def performance_vs_wellposedness(self, T21_true, T22_true, true, approx, error):
        """
        PERFORMANCE_VS_WELLPOSEDNESS Caculates errors and standard deviations as
          a function of how well-posed the original problem is. Well-posedness is
          quantified by the ratio T22/T21. The larger this ratio, the easier the 
          problem.
        
        INPUT:
        ------
        T21_true (1D Tensor) - 
        T22_true (1D Tensor) - 
        true (1D Tensor) - Length N_test. 
        approx (1D Tensor) -  Length  N_test. The k-th entry of approx
          is an approximation for the k-th entry of true.

        OUTPUT:
        -------
        """
        # For each sample, the ratio of its target T22 to its target T21.
        # Measures how well-posed the problem is.
        sample_well_posedness = T22_true/T21_true
        # Sequence of unique ratios, ordered increasingly
        list_of_ratios = torch.unique(sample_well_posedness, sorted=True)
        num_of_ratios  = len(list_of_ratios)

        # Initialize containers
        prediction_stdevs            = np.zeros(num_of_ratios)        
        prediction_error_means       = np.zeros(num_of_ratios)
        prediction_error_stdevs      = np.zeros(num_of_ratios)
        twoprediction_error_means  = np.zeros(num_of_ratios)
        twoprediction_error_stdevs = np.zeros(num_of_ratios)

        for (i, ratio) in enumerate(list_of_ratios):
            # For this ratio, its locations amongst the samples
            ratio_locs = torch.where(sample_well_posedness == ratio)[0]

            these_targets  = torch.zeros(len(ratio_locs))
            these_approxes = torch.zeros(len(ratio_locs))
            
            # samples corresponding to this ratio
            these_targets  = true[ratio_locs]
            these_approxes = approx[ratio_locs]

            # standard deviation in prediction
            prediction_stdevs[i] = torch.std(these_approxes)

            # average error and standard deviation of error
            prediction_error_means[i], prediction_error_stdevs[i] = self.sample_averaged_abs_error_stdev(these_targets,
                                                                                                         these_approxes)

            # average percent error and standard deviation of percent error
            if error == 'PE':
                twoprediction_error_means[i], twoprediction_error_stdevs[i] =self.sample_averaged_perc_error_stdev(these_targets, these_approxes)
            if error == 'SE':
                twoprediction_error_means[i], twoprediction_error_stdevs[i] =self.sample_averaged_squa_error_stdev(these_targets,these_approxes)
            if error == 'LPE':
                twoprediction_error_means[i], twoprediction_error_stdevs[i] =self.log_sample_averaged_perc_error_stdev(these_targets, these_approxes)
            if error == 'LSE':
                twoprediction_error_means[i], twoprediction_error_stdevs[i] =self.log_sample_averaged_squa_error_stdev(these_targets,these_approxes)

        return (list_of_ratios,
                prediction_stdevs,
                prediction_error_means, prediction_error_stdevs,
                twoprediction_error_means, twoprediction_error_stdevs)
    
    def PE_performance_vs_wellposedness(self, T21_true, T22_true, true, approx):
        """
        PERFORMANCE_VS_WELLPOSEDNESS Caculates errors and standard deviations as
          a function of how well-posed the original problem is. Well-posedness is
          quantified by the ratio T22/T21. The larger this ratio, the easier the 
          problem.
        
        INPUT:
        ------
        T21_true (1D Tensor) - 
        T22_true (1D Tensor) - 
        true (1D Tensor) - Length N_test. 
        approx (1D Tensor) -  Length  N_test. The k-th entry of approx
          is an approximation for the k-th entry of true.

        OUTPUT:
        -------
        """
        # For each sample, the ratio of its target T22 to its target T21.
        # Measures how well-posed the problem is.
        sample_well_posedness = T22_true/T21_true
        # Sequence of unique ratios, ordered increasingly
        list_of_ratios = torch.unique(sample_well_posedness, sorted=True)
        num_of_ratios  = len(list_of_ratios)

        # Initialize containers
        prediction_stdevs            = np.zeros(num_of_ratios)        
        prediction_error_means       = np.zeros(num_of_ratios)
        prediction_error_stdevs      = np.zeros(num_of_ratios)
        prediction_perc_error_means  = np.zeros(num_of_ratios)
        prediction_perc_error_stdevs = np.zeros(num_of_ratios)

        for (i, ratio) in enumerate(list_of_ratios):
            # For this ratio, its locations amongst the samples
            ratio_locs = torch.where(sample_well_posedness == ratio)[0]

            these_targets  = torch.zeros(len(ratio_locs))
            these_approxes = torch.zeros(len(ratio_locs))
            
            # samples corresponding to this ratio
            these_targets  = true[ratio_locs]
            these_approxes = approx[ratio_locs]

            # standard deviation in prediction
            prediction_stdevs[i] = torch.std(these_approxes)

            # average error and standard deviation of error
            prediction_error_means[i], prediction_error_stdevs[i] = self.sample_averaged_abs_error_stdev(these_targets,
                                                                                                         these_approxes)

            # average percent error and standard deviation of percent error
            prediction_perc_error_means[i], prediction_perc_error_stdevs[i] =self.sample_averaged_perc_error_stdev(these_targets,
                                                                                                                   these_approxes)

        return (list_of_ratios,
                prediction_stdevs,
                prediction_error_means, prediction_error_stdevs,
                prediction_perc_error_means, prediction_perc_error_stdevs)
    
    def SE_performance_vs_wellposedness(self, T21_true, T22_true, true, approx):
        """
        PERFORMANCE_VS_WELLPOSEDNESS Caculates errors and standard deviations as
          a function of how well-posed the original problem is. Well-posedness is
          quantified by the ratio T22/T21. The larger this ratio, the easier the 
          problem.
        
        INPUT:
        ------
        T21_true (1D Tensor) - 
        T22_true (1D Tensor) - 
        true (1D Tensor) - Length N_test. 
        approx (1D Tensor) -  Length  N_test. The k-th entry of approx
          is an approximation for the k-th entry of true.

        OUTPUT:
        -------
        """
        # For each sample, the ratio of its target T22 to its target T21.
        # Measures how well-posed the problem is.
        sample_well_posedness = T22_true/T21_true
        # Sequence of unique ratios, ordered increasingly
        list_of_ratios = torch.unique(sample_well_posedness, sorted=True)
        num_of_ratios  = len(list_of_ratios)

        # Initialize containers
        prediction_stdevs            = np.zeros(num_of_ratios)        
        prediction_error_means       = np.zeros(num_of_ratios)
        prediction_error_stdevs      = np.zeros(num_of_ratios)
        prediction_squa_error_means  = np.zeros(num_of_ratios)
        prediction_squa_error_stdevs = np.zeros(num_of_ratios)

        for (i, ratio) in enumerate(list_of_ratios):
            # For this ratio, its locations amongst the samples
            ratio_locs = torch.where(sample_well_posedness == ratio)[0]

            these_targets  = torch.zeros(len(ratio_locs))
            these_approxes = torch.zeros(len(ratio_locs))
            
            # samples corresponding to this ratio
            these_targets  = true[ratio_locs]
            these_approxes = approx[ratio_locs]

            # standard deviation in prediction
            prediction_stdevs[i] = torch.std(these_approxes)

            # average error and standard deviation of error
            prediction_error_means[i], prediction_error_stdevs[i] = self.sample_averaged_abs_error_stdev(these_targets,
                                                                                                         these_approxes)

            # average square error and standard deviation of square error
            prediction_perc_error_means[i], prediction_perc_error_stdevs[i] =self.sample_averaged_squa_error_stdev(these_targets,
                                                                                                                   these_approxes)

        return (list_of_ratios,
                prediction_stdevs,
                prediction_error_means, prediction_error_stdevs,
                prediction_squa_error_means, prediction_squa_error_stdevs)
    
    def LPE_performance_vs_wellposedness(self, T21_true, T22_true, true, approx):
        """
        PERFORMANCE_VS_WELLPOSEDNESS Caculates errors and standard deviations as
          a function of how well-posed the original problem is. Well-posedness is
          quantified by the ratio T22/T21. The larger this ratio, the easier the 
          problem.
        
        INPUT:
        ------
        T21_true (1D Tensor) - 
        T22_true (1D Tensor) - 
        true (1D Tensor) - Length N_test. 
        approx (1D Tensor) -  Length  N_test. The k-th entry of approx
          is an approximation for the k-th entry of true.

        OUTPUT:
        -------
        """
        # For each sample, the ratio of its target T22 to its target T21.
        # Measures how well-posed the problem is.
        sample_well_posedness = T22_true/T21_true
        # Sequence of unique ratios, ordered increasingly
        list_of_ratios = torch.unique(sample_well_posedness, sorted=True)
        num_of_ratios  = len(list_of_ratios)

        # Initialize containers
        prediction_stdevs            = np.zeros(num_of_ratios)        
        prediction_error_means       = np.zeros(num_of_ratios)
        prediction_error_stdevs      = np.zeros(num_of_ratios)
        log_prediction_perc_error_means  = np.zeros(num_of_ratios)
        log_prediction_perc_error_stdevs = np.zeros(num_of_ratios)

        for (i, ratio) in enumerate(list_of_ratios):
            # For this ratio, its locations amongst the samples
            ratio_locs = torch.where(sample_well_posedness == ratio)[0]

            these_targets  = torch.zeros(len(ratio_locs))
            these_approxes = torch.zeros(len(ratio_locs))
            
            # samples corresponding to this ratio
            these_targets  = true[ratio_locs]
            these_approxes = approx[ratio_locs]

            # standard deviation in prediction
            prediction_stdevs[i] = torch.std(these_approxes)

            # average error and standard deviation of error
            prediction_error_means[i], prediction_error_stdevs[i] = self.sample_averaged_abs_error_stdev(these_targets,
                                                                                                         these_approxes)

            # average log percent error and standard deviation of log percent error
            log_prediction_perc_error_means[i], log_prediction_perc_error_stdevs[i] =self.log_sample_averaged_perc_error_stdev(these_targets,
                                                                                                                   these_approxes)

        return (list_of_ratios,
                prediction_stdevs,
                prediction_error_means, prediction_error_stdevs,
                log_prediction_perc_error_means, log_prediction_perc_error_stdevs)
    
    def LSE_performance_vs_wellposedness(self, T21_true, T22_true, true, approx):
        """
        PERFORMANCE_VS_WELLPOSEDNESS Caculates errors and standard deviations as
          a function of how well-posed the original problem is. Well-posedness is
          quantified by the ratio T22/T21. The larger this ratio, the easier the 
          problem.
        
        INPUT:
        ------
        T21_true (1D Tensor) - 
        T22_true (1D Tensor) - 
        true (1D Tensor) - Length N_test. 
        approx (1D Tensor) -  Length  N_test. The k-th entry of approx
          is an approximation for the k-th entry of true.

        OUTPUT:
        -------
        """
        # For each sample, the ratio of its target T22 to its target T21.
        # Measures how well-posed the problem is.
        sample_well_posedness = T22_true/T21_true
        # Sequence of unique ratios, ordered increasingly
        list_of_ratios = torch.unique(sample_well_posedness, sorted=True)
        num_of_ratios  = len(list_of_ratios)

        # Initialize containers
        prediction_stdevs            = np.zeros(num_of_ratios)        
        prediction_error_means       = np.zeros(num_of_ratios)
        prediction_error_stdevs      = np.zeros(num_of_ratios)
        log_prediction_squa_error_means  = np.zeros(num_of_ratios)
        log_prediction_squa_error_stdevs = np.zeros(num_of_ratios)

        for (i, ratio) in enumerate(list_of_ratios):
            # For this ratio, its locations amongst the samples
            ratio_locs = torch.where(sample_well_posedness == ratio)[0]

            these_targets  = torch.zeros(len(ratio_locs))
            these_approxes = torch.zeros(len(ratio_locs))
            
            # samples corresponding to this ratio
            these_targets  = true[ratio_locs]
            these_approxes = approx[ratio_locs]

            # standard deviation in prediction
            prediction_stdevs[i] = torch.std(these_approxes)

            # average error and standard deviation of error
            prediction_error_means[i], prediction_error_stdevs[i] = self.sample_averaged_abs_error_stdev(these_targets,
                                                                                                         these_approxes)

            # average square error and standard deviation of square error
            log_prediction_squa_error_means[i], log_prediction_squa_error_stdevs[i] =self.log_sample_averaged_squa_error_stdev(these_targets,
                                                                                                                   these_approxes)

        return (list_of_ratios,
                prediction_stdevs,
                prediction_error_means, prediction_error_stdevs,
                log_prediction_squa_error_means, log_prediction_squa_error_stdevs)

    def performance_vs_T2s(self, T21_true, T22_true, true, approx, Error):
        """
        - Caculates average predictions, errors and standard deviations as a 
          function of target (T21,T22).

        - true and approx are 1D Tensors of length N_test. The k-th entry of approx
          is an approximation for the k-th entry of true.
        """
        T21_unique = torch.unique(T21_true, sorted=True)
        T22_unique = torch.unique(T22_true, sorted=True)
        num_T21, num_T22 = len(T21_unique), len(T22_unique)

        # Initialize containers
        prediction_means             = torch.zeros((num_T21, num_T22))
        prediction_stdevs            = torch.zeros((num_T21, num_T22))        
        prediction_error_means       = torch.zeros((num_T21, num_T22))
        prediction_error_stdevs      = torch.zeros((num_T21, num_T22))
        twoprediction_error_means  = torch.zeros((num_T21, num_T22))
        twoprediction_error_stdevs = torch.zeros((num_T21, num_T22))

        num_missing_pairs = 0
        # for visualizing which pairs (T21, T22) are missing in testing set
        
        for (j, T2) in enumerate(T22_unique):
            # For this T2, location of all samples with target T22==T2
            T2_locs = torch.where(T22_true == T2)[0]
            for (i, T1) in enumerate(T21_unique):
                # For this T1, location of all samples with target T22==T2 and target T21==T1
                T1_locs_from_T2_locs = torch.where(T21_true[T2_locs] == T1)[0]
                if T1_locs_from_T2_locs.nelement() == 0:
                    num_missing_pairs += 1
#                    print(f'no samples for (T21,T22)=({T1:.2f},{T2:.2f})')
                    continue

                these_targets = torch.zeros(len(T1_locs_from_T2_locs))
                these_approxes = torch.zeros(len(T1_locs_from_T2_locs))
                
                # samples corresponding to this ratio
                these_targets  = true[T2_locs][T1_locs_from_T2_locs]  
                these_approxes = approx[T2_locs][T1_locs_from_T2_locs]

                # mean prediction
                prediction_means[i,j] = torch.mean(these_approxes, dim=0)
                # standard deviation in prediction
                prediction_stdevs[i,j] = torch.std(these_approxes, dim=0)

                # average absolute error and standard deviation of error
                (prediction_error_means[i,j], prediction_error_stdevs[i,j]) = self.sample_averaged_abs_error_stdev(these_targets,
                                                                                                             these_approxes)

                # average specific error and standard deviation of error
                if Error == 'PE':
                    (twoprediction_error_means[i,j], twoprediction_error_stdevs[i,j]) =self.sample_averaged_perc_error_stdev(these_targets,these_approxes)
                
                if Error == 'SE':
                    (twoprediction_error_means[i,j], twoprediction_error_stdevs[i,j]) =self.sample_averaged_squa_error_stdev(these_targets,these_approxes)
                
                if Error == 'LPE':
                    (twoprediction_error_means[i,j], twoprediction_error_stdevs[i,j]) =self.log_sample_averaged_perc_error_stdev(these_targets,these_approxes)
                
                if Error == 'LSE':
                    (twoprediction_error_means[i,j], twoprediction_error_stdevs[i,j]) =self.log_sample_averaged_squa_error_stdev(these_targets,these_approxes)
                
                
                
#                print(f'(T21,T22)=({T1:.2f},{T2:.2f}), avg. %-error: {prediction_perc_error_means[i,j]:.3f}, %-error st.dev.: {prediction_perc_error_stdevs[i,j]:.3f}')
        print(f'number of missing pairs: {num_missing_pairs} of {num_T21*num_T22} ({100.0*np.float(num_missing_pairs)/np.float(num_T21*num_T22):.2f} %)')
        
        
        return (T21_unique, T22_unique,
                prediction_means, prediction_stdevs,
                prediction_error_means, prediction_error_stdevs,
                twoprediction_error_means, twoprediction_error_stdevs)


    def PE_performance_vs_T2s(self, T21_true, T22_true, true, approx):
        """
        - Caculates average predictions, errors and standard deviations as a 
          function of target (T21,T22).

        - true and approx are 1D Tensors of length N_test. The k-th entry of approx
          is an approximation for the k-th entry of true.
        """
        T21_unique = torch.unique(T21_true, sorted=True)
        T22_unique = torch.unique(T22_true, sorted=True)
        num_T21, num_T22 = len(T21_unique), len(T22_unique)

        # Initialize containers
        prediction_means             = torch.zeros((num_T21, num_T22))
        prediction_stdevs            = torch.zeros((num_T21, num_T22))        
        prediction_error_means       = torch.zeros((num_T21, num_T22))
        prediction_error_stdevs      = torch.zeros((num_T21, num_T22))
        prediction_perc_error_means  = torch.zeros((num_T21, num_T22))
        prediction_perc_error_stdevs = torch.zeros((num_T21, num_T22))

        num_missing_pairs = 0
        # for visualizing which pairs (T21, T22) are missing in testing set
        
        for (j, T2) in enumerate(T22_unique):
            # For this T2, location of all samples with target T22==T2
            T2_locs = torch.where(T22_true == T2)[0]
            for (i, T1) in enumerate(T21_unique):
                # For this T1, location of all samples with target T22==T2 and target T21==T1
                T1_locs_from_T2_locs = torch.where(T21_true[T2_locs] == T1)[0]
                if T1_locs_from_T2_locs.nelement() == 0:
                    num_missing_pairs += 1
#                    print(f'no samples for (T21,T22)=({T1:.2f},{T2:.2f})')
                    continue

                these_targets = torch.zeros(len(T1_locs_from_T2_locs))
                these_approxes = torch.zeros(len(T1_locs_from_T2_locs))
                
                # samples corresponding to this ratio
                these_targets  = true[T2_locs][T1_locs_from_T2_locs]  
                these_approxes = approx[T2_locs][T1_locs_from_T2_locs]

                # mean prediction
                prediction_means[i,j] = torch.mean(these_approxes, dim=0)
                # standard deviation in prediction
                prediction_stdevs[i,j] = torch.std(these_approxes, dim=0)

                # average error and standard deviation of error
                (prediction_error_means[i,j], prediction_error_stdevs[i,j]) = self.sample_averaged_abs_error_stdev(these_targets,
                                                                                                             these_approxes)

                # average percent error and standard deviation of percent error
                (prediction_perc_error_means[i,j], prediction_perc_error_stdevs[i,j]) =self.sample_averaged_perc_error_stdev(these_targets,
                                                                                                                       these_approxes)
#                print(f'(T21,T22)=({T1:.2f},{T2:.2f}), avg. %-error: {prediction_perc_error_means[i,j]:.3f}, %-error st.dev.: {prediction_perc_error_stdevs[i,j]:.3f}')
        print(f'number of missing pairs: {num_missing_pairs} of {num_T21*num_T22} ({100.0*np.float(num_missing_pairs)/np.float(num_T21*num_T22):.2f} %)')
        return (T21_unique, T22_unique,
                prediction_means, prediction_stdevs,
                prediction_error_means, prediction_error_stdevs,
                prediction_perc_error_means, prediction_perc_error_stdevs)
    
    def SE_performance_vs_T2s(self, T21_true, T22_true, true, approx):
        """
        - Caculates average predictions, errors and standard deviations as a 
          function of target (T21,T22).

        - true and approx are 1D Tensors of length N_test. The k-th entry of approx
          is an approximation for the k-th entry of true.
        """
        T21_unique = torch.unique(T21_true, sorted=True)
        T22_unique = torch.unique(T22_true, sorted=True)
        num_T21, num_T22 = len(T21_unique), len(T22_unique)

        # Initialize containers
        prediction_means             = torch.zeros((num_T21, num_T22))
        prediction_stdevs            = torch.zeros((num_T21, num_T22))        
        prediction_error_means       = torch.zeros((num_T21, num_T22))
        prediction_error_stdevs      = torch.zeros((num_T21, num_T22))
        prediction_squa_error_means  = torch.zeros((num_T21, num_T22))
        prediction_squa_error_stdevs = torch.zeros((num_T21, num_T22))

        num_missing_pairs = 0
        # for visualizing which pairs (T21, T22) are missing in testing set
        
        for (j, T2) in enumerate(T22_unique):
            # For this T2, location of all samples with target T22==T2
            T2_locs = torch.where(T22_true == T2)[0]
            for (i, T1) in enumerate(T21_unique):
                # For this T1, location of all samples with target T22==T2 and target T21==T1
                T1_locs_from_T2_locs = torch.where(T21_true[T2_locs] == T1)[0]
                if T1_locs_from_T2_locs.nelement() == 0:
                    num_missing_pairs += 1
#                    print(f'no samples for (T21,T22)=({T1:.2f},{T2:.2f})')
                    continue

                these_targets = torch.zeros(len(T1_locs_from_T2_locs))
                these_approxes = torch.zeros(len(T1_locs_from_T2_locs))
                
                # samples corresponding to this ratio
                these_targets  = true[T2_locs][T1_locs_from_T2_locs]  
                these_approxes = approx[T2_locs][T1_locs_from_T2_locs]

                # mean prediction
                prediction_means[i,j] = torch.mean(these_approxes, dim=0)
                # standard deviation in prediction
                prediction_stdevs[i,j] = torch.std(these_approxes, dim=0)

                # average error and standard deviation of error
                (prediction_error_means[i,j], prediction_error_stdevs[i,j]) = self.sample_averaged_abs_error_stdev(these_targets,
                                                                                                             these_approxes)

                # average square error and standard deviation of square error
                (prediction_squa_error_means[i,j], prediction_squa_error_stdevs[i,j]) =self.sample_averaged_squa_error_stdev(these_targets,
                                                                                                                       these_approxes)
#                print(f'(T21,T22)=({T1:.2f},{T2:.2f}), avg. %-error: {prediction_perc_error_means[i,j]:.3f}, %-error st.dev.: {prediction_perc_error_stdevs[i,j]:.3f}')
        print(f'number of missing pairs: {num_missing_pairs} of {num_T21*num_T22} ({100.0*np.float(num_missing_pairs)/np.float(num_T21*num_T22):.2f} %)')
        return (T21_unique, T22_unique,
                prediction_means, prediction_stdevs,
                prediction_error_means, prediction_error_stdevs,
                prediction_squa_error_means, prediction_squa_error_stdevs)
    
    def LPE_performance_vs_T2s(self, T21_true, T22_true, true, approx):
        """
        - Caculates average predictions, errors and standard deviations as a 
          function of target (T21,T22).

        - true and approx are 1D Tensors of length N_test. The k-th entry of approx
          is an approximation for the k-th entry of true.
        """
        T21_unique = torch.unique(T21_true, sorted=True)
        T22_unique = torch.unique(T22_true, sorted=True)
        num_T21, num_T22 = len(T21_unique), len(T22_unique)

        # Initialize containers
        prediction_means             = torch.zeros((num_T21, num_T22))
        prediction_stdevs            = torch.zeros((num_T21, num_T22))        
        prediction_error_means       = torch.zeros((num_T21, num_T22))
        prediction_error_stdevs      = torch.zeros((num_T21, num_T22))
        log_prediction_perc_error_means  = torch.zeros((num_T21, num_T22))
        log_prediction_perc_error_stdevs = torch.zeros((num_T21, num_T22))

        num_missing_pairs = 0
        # for visualizing which pairs (T21, T22) are missing in testing set
        
        for (j, T2) in enumerate(T22_unique):
            # For this T2, location of all samples with target T22==T2
            T2_locs = torch.where(T22_true == T2)[0]
            for (i, T1) in enumerate(T21_unique):
                # For this T1, location of all samples with target T22==T2 and target T21==T1
                T1_locs_from_T2_locs = torch.where(T21_true[T2_locs] == T1)[0]
                if T1_locs_from_T2_locs.nelement() == 0:
                    num_missing_pairs += 1
#                    print(f'no samples for (T21,T22)=({T1:.2f},{T2:.2f})')
                    continue

                these_targets = torch.zeros(len(T1_locs_from_T2_locs))
                these_approxes = torch.zeros(len(T1_locs_from_T2_locs))
                
                # samples corresponding to this ratio
                these_targets  = true[T2_locs][T1_locs_from_T2_locs]  
                these_approxes = approx[T2_locs][T1_locs_from_T2_locs]

                # mean prediction
                prediction_means[i,j] = torch.mean(these_approxes, dim=0)
                # standard deviation in prediction
                prediction_stdevs[i,j] = torch.std(these_approxes, dim=0)

                # average error and standard deviation of error
                (prediction_error_means[i,j], prediction_error_stdevs[i,j]) = self.sample_averaged_abs_error_stdev(these_targets,
                                                                                                             these_approxes)

                # average percent error and standard deviation of percent error
                (log_prediction_perc_error_means[i,j], log_prediction_perc_error_stdevs[i,j]) =self.log_sample_averaged_perc_error_stdev(these_targets,
                                                                                                                       these_approxes)
#                print(f'(T21,T22)=({T1:.2f},{T2:.2f}), avg. %-error: {prediction_perc_error_means[i,j]:.3f}, %-error st.dev.: {prediction_perc_error_stdevs[i,j]:.3f}')
        print(f'number of missing pairs: {num_missing_pairs} of {num_T21*num_T22} ({100.0*np.float(num_missing_pairs)/np.float(num_T21*num_T22):.2f} %)')
        return (T21_unique, T22_unique,
                prediction_means, prediction_stdevs,
                prediction_error_means, prediction_error_stdevs,
                log_prediction_perc_error_means, log_prediction_perc_error_stdevs)
    
    def LSE_performance_vs_T2s(self, T21_true, T22_true, true, approx):
        """
        - Caculates average predictions, errors and standard deviations as a 
          function of target (T21,T22).

        - true and approx are 1D Tensors of length N_test. The k-th entry of approx
          is an approximation for the k-th entry of true.
        """
        T21_unique = torch.unique(T21_true, sorted=True)
        T22_unique = torch.unique(T22_true, sorted=True)
        num_T21, num_T22 = len(T21_unique), len(T22_unique)

        # Initialize containers
        prediction_means             = torch.zeros((num_T21, num_T22))
        prediction_stdevs            = torch.zeros((num_T21, num_T22))        
        prediction_error_means       = torch.zeros((num_T21, num_T22))
        prediction_error_stdevs      = torch.zeros((num_T21, num_T22))
        log_prediction_squa_error_means  = torch.zeros((num_T21, num_T22))
        log_prediction_squa_error_stdevs = torch.zeros((num_T21, num_T22))

        num_missing_pairs = 0
        # for visualizing which pairs (T21, T22) are missing in testing set
        
        for (j, T2) in enumerate(T22_unique):
            # For this T2, location of all samples with target T22==T2
            T2_locs = torch.where(T22_true == T2)[0]
            for (i, T1) in enumerate(T21_unique):
                # For this T1, location of all samples with target T22==T2 and target T21==T1
                T1_locs_from_T2_locs = torch.where(T21_true[T2_locs] == T1)[0]
                if T1_locs_from_T2_locs.nelement() == 0:
                    num_missing_pairs += 1
#                    print(f'no samples for (T21,T22)=({T1:.2f},{T2:.2f})')
                    continue

                these_targets = torch.zeros(len(T1_locs_from_T2_locs))
                these_approxes = torch.zeros(len(T1_locs_from_T2_locs))
                
                # samples corresponding to this ratio
                these_targets  = true[T2_locs][T1_locs_from_T2_locs]  
                these_approxes = approx[T2_locs][T1_locs_from_T2_locs]

                # mean prediction
                prediction_means[i,j] = torch.mean(these_approxes, dim=0)
                # standard deviation in prediction
                prediction_stdevs[i,j] = torch.std(these_approxes, dim=0)

                # average error and standard deviation of error
                (prediction_error_means[i,j], prediction_error_stdevs[i,j]) = self.sample_averaged_abs_error_stdev(these_targets,
                                                                                                             these_approxes)

                # average square error and standard deviation of square error
                (log_prediction_squa_error_means[i,j], log_prediction_squa_error_stdevs[i,j]) =self.log_sample_averaged_squa_error_stdev(these_targets,
                                                                                                                       these_approxes)
#                print(f'(T21,T22)=({T1:.2f},{T2:.2f}), avg. %-error: {prediction_perc_error_means[i,j]:.3f}, %-error st.dev.: {prediction_perc_error_stdevs[i,j]:.3f}')
        print(f'number of missing pairs: {num_missing_pairs} of {num_T21*num_T22} ({100.0*np.float(num_missing_pairs)/np.float(num_T21*num_T22):.2f} %)')
        return (T21_unique, T22_unique,
                prediction_means, prediction_stdevs,
                prediction_error_means, prediction_error_stdevs,
                log_prediction_squa_error_means, log_prediction_squa_error_stdevs)
    
    