#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import sys
import pickle
import gc
from sklearn.metrics import root_mean_squared_error
import Testing_DeepSHAP_MPVI as nn
from scipy.stats import sobol_indices
from scipy.stats import norm

# In[ ]:
#this function calculates the sobol indices
def get_sobol_scipy(input_shape, loc, scale, n, func):
    indices = sobol_indices(func=func,
                        n=n,
                        dists=[norm(loc = loc, scale = scale) for i in range(input_shape)],
                        random_state=13)
    return indices.first_order, indices.total_order, indices.total_order - indices.first_order

#bootstrapping routine for VBSA for the testing on synthetic data
def bootstrap_routine(input_shape, epochs, y_func, test, noise, noise_level, n_bootstraps, path, N,  num_parallel_runs=10):
    # create filename
    filename = f'{y_func}_{test}_{noise}_{noise_level}_'
    #create pkl files to save the data in later
    with open(path + filename + 'history.pkl', 'wb') as f:
        pickle.dump([], f)
    with open(path + filename + 'sobol.pkl', 'wb') as f:
        pickle.dump([], f)
    #define the tuned parameters in a dictionary for the synthetic data
    if y_func == 'polynom':
        param_dict = {'input_shape': input_shape, 'dropout': 0.2, 'act_func': 'relu', 'optimizer': 'adam',
                'learning_rate': 0.01, 'regularizer_choice': 'None', 'num_neurons_0': 100, 'name': 'NN'}
    if y_func == 'non_additive':
        param_dict = {'input_shape': input_shape, 'dropout': 0.0, 'act_func': 'sigmoid', 'optimizer': 'adam',
                'learning_rate': 0.01, 'regularizer_choice': 'None', 'num_neurons_0': 80, 'name': 'NN'}
    #get the training and test sets of the synthetic data
    X_train, X_test, y_train, y_test = nn.get_data(y_func, test, noise, noise_level, size = 5000, seed = 42)

    #iterate over N bootstraps
    for i in range(n_bootstraps):
        #we draw as many samples as the training set size with replacement and create our bootstrap sample
        ind = np.random.choice(np.arange(X_train.shape[0]), size = X_train.shape[0])
        X_train_boot = X_train[ind]
        y_train_boot = y_train[ind]
        #we build the tuned model
        model = nn.build_model_NN(param_dict)
        #train the model on the bootstrap sample
        history = model.fit(X_train_boot, y_train_boot, epochs=epochs, batch_size=X_train.shape[0], validation_split=0.2, verbose = 0)
        #calculate performance of the NN on the test set
        y_pred = model.predict(X_test)
        res = root_mean_squared_error(y_test, y_pred)
        #define the predictive function
        def predict_fn(X):
            X = X.transpose()
            return model.predict(X).transpose()
        #calculate the sobol indices
        Sfirst, Stot, Sinter = get_sobol_scipy(input_shape, 0, 1, N, func = predict_fn)
        #save the results in the pkl files
        with open(path + filename + 'history.pkl', 'rb') as f:
            h = pickle.load(f)
        h.append(res)
        print(h)
        with open(path + filename + 'history.pkl', 'wb') as f:
            pickle.dump(h, f)
        
        with open(path + filename + 'sobol.pkl', 'rb') as f:
            s = pickle.load(f)
        s.append([Sfirst, Stot, Sinter])
        print(s)
        with open(path + filename + 'sobol.pkl', 'wb') as f:
            pickle.dump(s, f)
        #collect garbadge and print iteration step 
        gc.collect()
        sys.stdout.write('\riteration step %d' %i)
        sys.stdout.flush()