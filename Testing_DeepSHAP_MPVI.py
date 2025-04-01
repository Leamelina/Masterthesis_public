import numpy as np
import sys
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import shap
import gc
from sklearn.metrics import root_mean_squared_error
import sys
import PermutationImportance

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
    dropout_rate = param_dict['dr']
    activation_func = param_dict['act_func']
    optimizer_choice = param_dict['optimizer']
    learning_rate = param_dict['learning_rate']
    num_layers = 1
    regularizer_choice = param_dict['regularizer_choice']
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
        num_neurons = param_dict['num_neurons_0']
        x = block(x, num_neurons, activation_func, regularizer, dropout_rate)

    # output is the output layer which has a linear activation function
    outputs = layers.Dense(1, activation='linear')(x)
    # here we create the model defined above, which is now called model
    model = tf.keras.Model(inputs, outputs, name=param_dict['name'])

    # the optimizer is set to adam or SGD with the learning rate defined in the parameter dictionary 
    # the model is then compiled, the loss function set to the MSE and the performance metrics on the validation set during training to MAE
    if optimizer_choice == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model

#this function trains the model
def train_model(param_dict, y_func, epochs, eval_func, seed, test, noise, noise_level, size, verbose = 0):
    #build the model
    model = build_model_NN(param_dict)
    #slpit the data into training and test set
    X_train, X_test, y_train, y_test =  get_data(y_func, test, noise, noise_level, size = size, seed = seed)
    #train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=X_train.shape[0], validation_split=0.2, verbose = verbose)
    #calculate performance on the test set
    y_pred = model.predict(X_test)
    res = eval_func(y_test, y_pred)
    return model, history, res, X_train, y_train, X_test, y_test

#this function parallelizes the training process
def parallel_train(param_dicts,y_func, epochs, seed, test, size, noise, noise_level, eval_func = root_mean_squared_error, num_parallel_runs=10):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(train_model, pd, y_func, epochs, eval_func, seed+i, test, noise, noise_level, size) for i, pd in enumerate(param_dicts)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    return results


#bootstrap over NN trained on the synthetic test data
def Bootstrap_SHAP_testing(parameter_dict, path, N, Epochs, y_func, test, noise, noise_level, size = 5000, num_parallel_runs=10, new = True, seed=42):
    s = seed
    if noise_level == 1.0:
        nl = '1'
    if noise_level == 0.5:
        nl = '05'
    if noise_level == 0.1:
        nl = '01'
    else:
        nl = 'None'
    #if this is a new run, create a pkl file in which the resluts, i.e. the SHAP vaöues and performance of the NN, will be saved
    if new == True:
        with open(path + 'SHAP_test_history_' + y_func +  test +'_' + noise + nl+'.pkl', 'wb') as f:
            pickle.dump([], f)
        with open(path + 'SHAP_test_shap_values_' + y_func + test +'_' + noise + nl+'.pkl', 'wb') as f:
            pickle.dump([], f)
        with open(path + 'SHAP_test_data_' + y_func  + test+'_' + noise + nl+ '.pkl', 'wb') as f:
            pickle.dump([], f)
    #we parallelize this on CPUs therefore we need to iterate over N/num_parallel_runs
    for i in range(int(N/num_parallel_runs)):
        param_dict_list = []
        #create num_parallel_runs parameter dictionaries
        for n in range(num_parallel_runs):
            temp_dict = parameter_dict.copy()
            temp_dict['name'] = f'NN_{i}'
            param_dict_list.append(temp_dict)
        #train model with the function parallel_train from the NN_tuned_test notebook
        results = parallel_train(param_dict_list, y_func, Epochs, s, test, 5000, noise, noise_level, eval_func = root_mean_squared_error, num_parallel_runs=num_parallel_runs)
        #save the performance history of the NNs
        with open(path + 'SHAP_test_history_' + y_func + test +'_' + noise + nl+ '.pkl', 'rb') as f:
            history_list = pickle.load(f)
        histories = [[r[1].history, r[2]] for r in results]
        history_list.append([histories])
        with open(path + 'SHAP_test_history_' + y_func +  test +'_' + noise + nl+'.pkl', 'wb') as f:
            pickle.dump(history_list, f)
        #save the test data 
        with open(path + 'SHAP_test_data_' + y_func +  test +'_' + noise + nl+'.pkl', 'rb') as f:
            data_list = pickle.load(f)
        datas = [[r[3], r[4], r[5], r[6]] for r in results]
        data_list.append([datas])
        with open(path + 'SHAP_test_data_' + y_func +  test +'_' + noise + nl+'.pkl', 'wb') as f:
            pickle.dump(data_list, f)

        models = [r[0] for r in results]
        X_trains = [d[0] for d in datas]
        X_tests = [d[2] for d in datas]
        print(len(X_trains))
        print(X_trains[0].shape)
        print(X_tests[0].shape)
        #draw 1000 samples with replacement from the training set, and do this num_parallel_runs time
        inds = [np.random.choice(np.arange(X_trains[0].shape[0]), size = 1000, replace = False) for i in range(num_parallel_runs)]
        #set up the num_parallel_runs explainer with the backgroundsamples
        exp_SHAPs = [shap.DeepExplainer(model, X_train[ind]) for model, ind, X_train in zip(models, inds, X_trains)]
        #calculate the SHAP values for all of them
        shap_values = [exp_SHAP(X_test) for exp_SHAP, X_test in zip(exp_SHAPs, X_tests)]
        #add them to the pkl file
        with open(path + 'SHAP_test_shap_values_' + y_func +  test +'_' + noise + nl+'.pkl', 'rb') as f:
            shap_values_list = pickle.load(f)
        shap_values_list.append(shap_values)
        with open(path + 'SHAP_test_shap_values_' + y_func +  test +'_' + noise + nl+ '.pkl', 'wb') as f:
            pickle.dump(shap_values_list, f)
        
        #collect garbadge and print iteration step
        s+=num_parallel_runs
        gc.collect()
        sys.stdout.write('\riteration step %d' %i)
        sys.stdout.flush()
#this function creates synthetic testing and training data
def get_data(y_func, test, noise, noise_level, size = 5000, seed = 42):
    if noise_level == 1.0:
        nl = '1'
    if noise_level == 0.5:
        nl = '05'
    if noise_level == 0.1:
        nl = '01'
    else:
        nl = 'None'
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
    #split the data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size = 0.33, random_state = seed)
    return X_train, X_test, y_train, y_test

#routine for bootstrapping DeepSHAP and MPVI on the test data
def routine(y_func, test, noise, noise_level, size = 5000, new = True, N=100):
    if noise_level == 1.0:
        nl = '1'
    if noise_level == 0.5:
        nl = '05'
    if noise_level == 0.1:
        nl = '01'
    else:
        nl = 'None'
    print(noise_level)
    #create parameter dictionary with the tuned parameters for the two test functions
    if y_func == 'polynom':
        param_dict = {'input_shape': 5, 'dr': 0.2, 'act_func': 'relu', 'optimizer': 'adam',
                      'learning_rate': 0.01, 'regularizer_choice': 'None', 'num_neurons_0': 100, 'name': 'NN'}
    if y_func == 'non_additive':
        param_dict = {'input_shape': 5, 'dr': 0.0, 'act_func': 'sigmoid', 'optimizer': 'adam',
                      'learning_rate': 0.01, 'regularizer_choice': 'None', 'num_neurons_0': 80, 'name': 'NN'}
    #boostrap the SHAP values
    Bootstrap_SHAP_testing(param_dict, 'Data/SHAP/test/', N, 5000, y_func, test, noise, noise_level, size = 5000, num_parallel_runs=10, new = new)
    #get testing and training data
    X_train, X_test, y_train, y_test = get_data(y_func, test, noise, noise_level)
    #train the model
    model, history, result = train_model(param_dict, X_train, y_train, X_test, y_test, 5000, root_mean_squared_error)
    #save the model history in the pkl file
    with open('Data/PVI/test/Model_'+ y_func + test +'_' + noise + str(noise_level)+'.pkl', 'wb') as f:
        pickle.dump([model, history.history, result], f)
    #define the scoring data
    scoring_data = (X_test, y_test)
    #create a list with the feature names
    predictor_names = np.array([f'x{i}' for i in range(1, X_train.shape[1]+1)])
    #use MPVI from the PermutationImportance packadge; the scoring function is RMSE; this bootstraps automatically in parallel
    result_RMSE = PermutationImportance.sklearn_permutation_importance(model, scoring_data, root_mean_squared_error, 'argmax_of_mean',
    variable_names=predictor_names, nbootstrap=100, subsample=1, nimportant_vars=5, njobs = 20)
    #save results in pkl file
    with open('Data/PVI/test/result_RMSE_'+ y_func+ test +'_' + noise + nl+ '.pkl', 'wb') as f:
        pickle.dump(result_RMSE, f)