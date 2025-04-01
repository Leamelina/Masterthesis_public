#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import concurrent.futures
from sklearn.preprocessing import StandardScaler
import shap
import scipy.stats as st
import sys
import pickle
import gc

#define the GELU activation function, as DeepSHAP can't calculate the gradients
def gelu_approx_no_pow(x):
    coeff = tf.constant(0.044715)
    sqrt_pi_over_2 = tf.sqrt(2 / tf.constant(math.pi))
    return 0.5 * x * (1 + tf.math.tanh(sqrt_pi_over_2 * x * (1 + coeff * x * x)))

#define a block of the NN
def block(pl, num_neurons, act_func, regularizer, dropout_rate):
    #l is one dense hidden layer
    l = layers.Dense(num_neurons, activation=act_func, kernel_regularizer=regularizer)(pl)
    #do is the dropout layer, if dropout_rate is set to 0, then nothing happens here
    do = layers.Dropout(dropout_rate)(l)
    return do

#this function builds the neural network model
def build_model_NN(param_dict):
    #inputs is the input layer, it's shape is defined in the parameter dictonary
    inputs = layers.Input(shape=(param_dict['input_shape'],))  # Assuming input_shape is defined elsewhere
    x = inputs

    # The tuned hyperparameters are defined in the parameter dictionary
    dropout_rate = param_dict['dropout_rate']
    activation_func = param_dict['activation_func']
    learning_rate = param_dict['learning_rate']
    num_layers = param_dict['num_layers']
    regularizer_choice = param_dict['kernel_regularizer']
    if regularizer_choice == 'l1':
        regularizer = tf.keras.regularizers.l1()
    elif regularizer_choice == 'l2':
        regularizer = tf.keras.regularizers.l2()
    elif regularizer_choice == 'l1_l2':
        regularizer = tf.keras.regularizers.l1_l2()
    else:
        regularizer = None

    # we stack as many blocks as there are hidden layers defined in the parameter dictionary, every hidden layer has it's own number of neurons defined
    for i in range(num_layers):
        num_neurons = param_dict[f'num_neurons_{i}']
        x = block(x, num_neurons, activation_func, regularizer, dropout_rate)

    # output is the output layer which has a linear activation function
    outputs = layers.Dense(1, activation='linear')(x)

    # here we create the model defined above, which is now called model
    model = tf.keras.Model(inputs, outputs, name=param_dict['name'])

    # the optimizer is set to adam with the learning rate defined in the parameter dictionary 
    # the model is then compiled, the loss function set to the MSE and the performance metrics on the validation set during training to MAE
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model

#this function calculates the Sobol' indices
def get_sobol_scipy(model, distr_par, Ns):
    #setup uniform distribution for each feature value
    dists = [st.uniform(loc=np.array(distr_par)[i, 0], scale=np.array(distr_par)[i, 1])for i in range(np.array(distr_par).shape[0])]
    #set up the predictive funtion, which is the NN
    def predict_fn(X):
        X = X.transpose()
        return model.predict(X).transpose()
    #calculate the sobol indices
    indices = st.sobol_indices(func=predict_fn, n=Ns, dists=dists)
    #return first-order, total-order and interaction sobol' indices
    return indices.first_order, indices.total_order, indices.total_order - indices.first_orde

def bootstrap_SHAP(parameter_dict, df_inputs, df_output, domain, output_name, N, epochs, locations, Ns = 8192, new = True, seed=42, local = False):
    #define a path where the results will be saved
    path = '/content/drive/MyDrive/Masterarbeit/SHAP_VBSA_data/'
    #standardize the data, so the NN can learn better and we prevent problems of exploding and vanishing gradients
    scaler = StandardScaler()
    df_inputs_standardized = pd.DataFrame(scaler.fit_transform(df_inputs), columns=df_inputs.columns, index = df_inputs.index)
    #split the data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(df_inputs_standardized, df_output, test_size = 0.2, random_state = seed)
    domain_name = domain[0]
    inputs_domain = domain[1]

    #if this is a new run, create a pkl file, where the results will be saved in
    if new == True:
        with open(path + 'Importances_res_' + output_name + '_' + domain_name + '_subset.pkl', 'wb') as f:
            pickle.dump([], f)
        with open(path + 'Importances_shap_values_'+ output_name + '_'+ domain_name +'_subset.pkl', 'wb') as f:
            pickle.dump([], f)
        if local == True:
            with open(path + 'Importances_VBSA_'+ output_name + '_'+ domain_name +'_subset_local.pkl', 'wb') as f:
                pickle.dump([], f)
        else:
            with open(path + 'Importances_VBSA_'+ output_name + '_'+ domain_name +'_subset.pkl', 'wb') as f:
                pickle.dump([], f)
        with open(path + 'Importances_ypred_' + output_name + '_'+ domain_name +'_subset.pkl', 'wb') as f:
            pickle.dump([], f)
        
    #loop over N bootstrap iterations
    for i in range(N):
        #build NN with the function build_model_NN
        model = build_model_NN(parameter_dict)
        #sample bootstrap set by drawing as many samples as there are in the training set with replacement
        ind = np.random.choice(np.arange(X_train.shape[0]), size = X_train.shape[0], replace = True)
        X_train_boot = X_train.iloc[ind]
        y_train_boot = y_train.iloc[ind]
        #train model on the bootstrap set
        history = model.fit(X_train_boot, y_train_boot, epochs=epochs, batch_size=X_train_boot.shape[0], validation_split=0.2, verbose = 0)
        #extract all samples in the test set that are in the domain we want to calculate the importances for
        idx_test_domain = X_test.index.intersection(inputs_domain)
        X_test_domain = X_test.loc[idx_test_domain]
        y_test_domain = y_test.loc[idx_test_domain]

        #predict the output for all data points that are in the domain
        y_pred = model.predict(X_test_domain)
        #calculate the performance over the domain
        RMSE = np.sqrt(mean_squared_error(y_test_domain, y_pred))
        R2 = r2_score(y_test_domain, y_pred)
        #sample the background set for the DeepSHAP
        inds_shap = np.random.choice(np.arange(X_train_boot_domain.shape[0]), size = 1000, replace = False)
        #setup the explainer with the background set on the model
        exp_SHAP = shap.DeepExplainer(model, np.array(X_train_boot_domain.iloc[inds_shap]))
        #calculate the SHAP values for the test samples within the domain
        shap_values = exp_SHAP(np.array(X_test_domain))

        #this calculates local VBSA values
        if local_VBSA == True:
            S = np.zeros((len(locations), X_test_domain.shape[1], 3))
            #loop over all locations local VBSA values should be calculated
            for xi, l in enumerate(locations):
                #the range of the uniform distribution for every value is set to [0.7*feature value; 1.3*feature value]
                minimums_local = X_test_domain.loc[l, :]*0.7
                maximums_local = X_test_domain.loc[l, :]*1.3
                distr_par_local= [[np.min(np.array([min_val, max_val])), np.abs((max_val - min_val))] for min_val, max_val in zip(minimums_local, maximums_local)]
                #calculate local VBSA values
                Sfirst, Stot, Sinter = get_sobol_scipy(model, distr_par_local, Ns = Ns)
                S[xi, :, 0] = Sfirst
                S[xi, :, 1] = Stot
                S[xi, :, 2] = Sinter

        #this calculates domain VBSA values
        else:
            S = np.zeros(X_test_domain.shape[1], 3)
            #the range of the uniform distribution for every value is set to 
            #[minimum feature value within the domain; maximum feature value within the domain]
            minimums_domain = np.min(X_test_domain, axis = 0)
            maximums_domain = np.max(X_test_domain, axis = 0)
            distr_par_domain = [[min_val, np.abs((max_val - min_val))] for min_val, max_val in zip(minimums_domain, maximums_domain)]
            #calculate domain VBSA values
            Sfirst, Stot, Sinter = get_sobol_scipy(model, distr_par_domain, Ns = Ns)
            S[:, 0] = Sfirst
            S[:, 1] = Stot
            S[:, 2] = Sinter

        #save the performances, SHAP values and VBSA values of this bootstrap iteration
        with open(path + 'Importances_res_' + output_name + '_' + domain_name + '_subset.pkl', 'rb') as f:
            res_list = pickle.load(f)
        res_list.append([RMSE, R2])
        print(R2)
        with open(path + 'Importances_res_' + output_name + '_' + domain_name + '_subset.pkl', 'wb') as f:
            pickle.dump(res_list, f)

        with open(path + 'Importances_ypred_' + output_name + '_'+ domain_name +'_subset.pkl', 'rb') as f:
            ypred_list = pickle.load(f)
        ypred_list.append(y_pred)
        with open(path + 'Importances_ypred_' + output_name + '_'+ domain_name +'_subset.pkl', 'wb') as f:
            pickle.dump(ypred_list, f)

        with open(path + 'Importances_shap_values_'+ output_name + '_'+ domain_name +'_subset.pkl', 'rb') as f:
            shap_values_list = pickle.load(f)
        shap_values_list.append(shap_values)
        with open(path + 'Importances_shap_values_'+ output_name + '_'+ domain_name +'_subset.pkl', 'wb') as f:
            pickle.dump(shap_values_list, f)

        if local == True:
            with open(path + 'Importances_VBSA_'+ output_name + '_'+ domain_name +'_subset_local.pkl', 'rb') as f:
                VBSA_list_local = pickle.load(f)
            VBSA_list_local.append([S])
            with open(path + 'Importances_VBSA_'+ output_name + '_'+ domain_name +'_subset_local.pkl', 'wb') as f:
                pickle.dump(VBSA_list_local, f)
        else:
            with open(path + 'Importances_VBSA_'+ output_name + '_'+ domain_name +'_subset.pkl', 'rb') as f:
                VBSA_list_local = pickle.load(f)
            VBSA_list_local.append([S])
            with open(path + 'Importances_VBSA_'+ output_name + '_'+ domain_name +'_subset.pkl', 'wb') as f:
                pickle.dump(VBSA_list_local, f)
        with open(path + 'Importances_VBSA_'+ output_name + '_'+ domain_name +'_subset.pkl', 'rb') as f:
            VBSA_list_global = pickle.load(f)
        VBSA_list_global.append([np.mean(S, axis = 0)])
        with open(path + 'Importances_VBSA_'+ output_name + '_'+ domain_name +'_subset.pkl', 'wb') as f:
            pickle.dump(VBSA_list_global, f)

        #collect garbadge
        gc.collect()
        #print iteration step
        sys.stdout.write('\riteration step %d' %i)
        sys.stdout.flush()
