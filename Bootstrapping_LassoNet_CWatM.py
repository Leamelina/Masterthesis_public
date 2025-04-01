#!/usr/bin/env python
# coding: utf-8

# In[ ]:

### Lizenzhinweis  
#Dieses Notebook verwendet die Bibliothek **[lassonet]**, die unter der MIT-Lizenz steht.  
#Copyright 2020 Louis Abraham, Ismael Lemhadri veröffentlicht mit der MIT-Lizenz 

import numpy as np 
from sklearn.model_selection import train_test_split
import lassonet
import pickle
import gc
import sys
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

def get_LassoNet_stats(X, y, output, domain, hyperparameters, filepath, n_iters=(1000, 100), M=10, N=100, new=True):

    hidden_dims = hyperparameters['hidden_dims']
    dropout = hyperparameters['dropout_rate']
    optim = hyperparameters['optim']

    #standardize the data, this helps the NN to learn and can prevent the exploding or vanishing gradients problem
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    #split the data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size = 0.2, random_state = 42)

    #if this is a new run, create a pkl file in which the resluts, i.e. the path and performance of LassoNet, will be saved
    if new == True:
        with open(filepath + 'LassoNet_' + output + '_' + domain + '_path.pkl', 'wb') as f:
            pickle.dump([], f)
        with open(filepath + 'LassoNet_' + output + '_' + domain + '_r2.pkl', 'wb') as f:
            pickle.dump([], f)
        with open(filepath + 'LassoNet_' + output + '_' + domain + '_rmse.pkl', 'wb') as f:
            pickle.dump([], f)

    #loop with N bootstrap iterations
    for j in range(N):
        #sample as many data points as the shape of the training set with replacement, this is our bootstrap set
        inds = np.random.choice(np.arange(X_train.shape[0]), size = X_train.shape[0], replace = True)
        X_train_boot = X_train[inds]
        y_train_boot = y_train[inds]

        #define the NN architecture for LassoNet using the tuned hyperparameters
        model = lassonet.LassoNetRegressor(hidden_dims = hidden_dims, M=M, batch_size = X_train_boot.shape[0], dropout = dropout, optim = optim, n_iters = n_iters)
        #train the LassoNet path on the bootstrap set
        path = model.path(X_train_boot, y_train_boot, return_state_dicts=True)
        
        #open the pkl file of the LassoNet path
        with open(filepath + 'LassoNet_' + output + '_' + domain + '_path.pkl', 'rb') as f:
            path_list = pickle.load(f)
        #append the new path to the list
        path_list.append(path)
        #save the new list, which includes the new bootstrap path
        with open(filepath + 'LassoNet_' + output + '_' + domain + '_path.pkl', 'wb') as f:
            pickle.dump(path_list, f)
        
        #create a list for the r2 and rmse score and append the performances of the LassoNet path on the test set
        test_score_r2_i = []
        test_score_rmse_i = []
        for save in path:
            model_lambda = model.load(save.state_dict)
            y_pred = model_lambda.predict(X_test)
            test_score_r2_i.append(r2_score(y_test, y_pred))
            test_score_rmse_i.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            
        #open the pkl file of the LassoNet R2 performances on the test set
        with open(filepath + 'LassoNet_' + output + '_' + domain + '_r2.pkl', 'rb') as f:
            test_score_r2 = pickle.load(f)
        #append the new R2 performances
        test_score_r2.append(test_score_r2_i)
        #save the new list, which includes the new R2 performances
        with open(filepath + 'LassoNet_' + output + '_' + domain +  '_r2.pkl', 'wb') as f:
            pickle.dump(test_score_r2, f)

        #open the pkl file of the LassoNet RMSE performances on the test set    
        with open(filepath + 'LassoNet_' + output + '_' + domain + '_rmse.pkl', 'rb') as f:
            test_score_rmse = pickle.load(f)
        #append the new RMSE performances
        test_score_rmse.append(test_score_rmse_i)
        #save the new list, which includes the new RMSE performances
        with open(filepath + 'LassoNet_' + output + '_' + domain + '_rmse.pkl', 'wb') as f:
            pickle.dump(test_score_rmse, f)
            
        #collect garbadge
        gc.collect()
        #print iteration step
        sys.stdout.write('\riteration step %d' %j)
        sys.stdout.flush()