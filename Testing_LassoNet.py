#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
from sklearn.model_selection import train_test_split
import lassonet
from sklearn.metrics import mean_squared_error
import pickle
import gc
import sys
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
# In[ ]:

#bootstrap over LassoNet trained on the synthetic test data
def bootstrap_LassoNet_testing(y_func, test, noise, noise_level, seed, M, path_to_file, N, size, hidden_dims, dropout, optim, preprocessing =False, new = False):
    n=seed
    #if this is a new run, create a pkl file in which the resluts, i.e. the path and performance of LassoNet, will be saved
    if new == True:
        with open(path_to_file + 'lassonet' + y_func +  test + '_' + noise + str(noise_level) + str(preprocessing) +'.pkl', 'wb') as f:
            pickle.dump([], f)
        with open(path_to_file + 'lassonet' + y_func +  test + '_' + noise + str(noise_level) + str(preprocessing) +'_testerror_r2.pkl', 'wb') as f:
            pickle.dump([], f)
        with open(path_to_file + 'lassonet' + y_func +  test + '_' + noise + str(noise_level) + str(preprocessing) +'_testerror_mse.pkl', 'wb') as f:
            pickle.dump([], f)
    #iterate over N bootstrap samples
    for j in range(N):
        #create 10 random gaussian variables
        for i in range(1, 11):
            rng = np.random.default_rng(seed+i)
            globals()[f"x{i}"] = rng.normal(size = size).reshape(size, 1)
        #we create 2 different test functions, a polynomial and a non-additive function
        if y_func == 'polynom':
            y = x1**5 + x2**4 + x3**3 + x4**2 + x5
        if y_func == 'non_additive':
            y = np.tanh(x3)*(np.sin(x1)*np.log(x2**2) + x4) +x5
        #we have three different tests:
        #add irrelevant features, i.e. thee five additinal random gaussian variable we created above
        if test == 'random_noise':
            data = np.concatenate((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10), axis = 1)
        #add noise to feature x1 and x3; here we have two options, either add gaussian noise or laplacian noise
        if test == 'noisy_data':
            rng_n = np.random.default_rng(seed+i+20)
            if noise == 'gaussian':
                gn = rng_n.normal(loc=0.0, scale=noise_level, size = size).reshape(size, 1)
                data = np.concatenate((x1+gn, x2, x3+gn, x4, x5), axis = 1)
            if noise == 'laplace':
                ln = rng_n.laplace(loc=0.0, scale = noise_level, size = size).reshape(size, 1)
                data = np.concatenate((x1+ln, x2, x3+ln, x4, x5), axis = 1)
        #add redundant features; these features have variying levels of correlation
        if test == 'redundant_data':
            rng_n = np.random.default_rng(seed+i+20)
            xn = rng_n.normal(loc=0.0, scale=1.0, size = size).reshape(size, 1)
            gn = rng_n.normal(loc=0.0, scale=noise_level, size = size).reshape(size, 1)
            data = np.concatenate((x1, x2, x3, x4, x5, 0.3*x1, 0.7*x1, x2/xn, xn, 0.7*x1+gn), axis = 1) 
        #create list with feature names
        feature_names = [f"x{i}" for i in range(1, data.shape[1]+1)]
        #prepreocess the data if needed by standardizing it; this can help the NN learn better
        if preprocessing == True:
            scaler = StandardScaler()
            data = pd.DataFrame(scaler.fit_transform(data), columns=feature_names)
        #split the data into training and test data
        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size = 0.33, random_state = n)
        #build the lassonet model
        model = lassonet.LassoNetRegressor(hidden_dims = hidden_dims, M=M, dropout = dropout, optim = optim)
        #train the lassonet path
        path = model.path(X_train, y_train, return_state_dicts=True)
        #save the results in the pkl file
        with open(path_to_file + 'lassonet' + y_func +  test + '_' + noise + str(noise_level) + str(preprocessing) +'.pkl', 'rb') as f:
            path_list = pickle.load(f)
        path_list.append(path)
        with open(path_to_file + 'lassonet' + y_func +  test + '_' + noise + str(noise_level) + str(preprocessing) +'.pkl', 'wb') as f:
            pickle.dump(path_list, f)
        
        #write the test scores of the path within a list
        test_score_r2_i = []
        test_score_mse_i = []
        for save in path:
            model_lambda = model.load(save.state_dict)
            y_pred = model_lambda.predict(X_test)
            test_score_r2_i.append(r2_score(y_test, y_pred))
            test_score_mse_i.append(mean_squared_error(y_test, y_pred))
        
        #save the test scores in the pkl files
        with open(path_to_file + 'lassonet' + y_func +  test + '_' + noise + str(noise_level) + str(preprocessing) + '_testerror_r2.pkl', 'rb') as f:
            test_score_r2 = pickle.load(f)
        test_score_r2.append(test_score_r2_i)
        with open(path_to_file + 'lassonet' + y_func +  test + '_' + noise + str(noise_level) + str(preprocessing) +'_testerror_r2.pkl', 'wb') as f:
            pickle.dump(test_score_r2, f)
            
        with open(path_to_file + 'lassonet' + y_func +  test + '_' + noise + str(noise_level) + str(preprocessing) +'_testerror_mse.pkl', 'rb') as f:
            test_score_mse = pickle.load(f)
        test_score_mse.append(test_score_mse_i)
        with open(path_to_file + 'lassonet' + y_func +  test + '_' + noise + str(noise_level) + str(preprocessing) +'_testerror_mse.pkl', 'wb') as f:
            pickle.dump(test_score_mse, f)
            
        #collect garbadge and print iteration step
        gc.collect()
        n+=1
        sys.stdout.write('\riteration step %d' %j)
        sys.stdout.flush()