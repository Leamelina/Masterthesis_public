#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import keras
from keras import layers
import keras_tuner
import numpy as np
from sklearn.model_selection import KFold
import keras_tuner

#define keras hyperparameter tuner
hp = keras_tuner.HyperParameters()

#define a block of the NN
def block(pl, num_neurons, act_func, regularizer, dropout_rate):
    #l is one dense hidden layer
    l = layers.Dense(num_neurons, activation=act_func, kernel_regularizer=regularizer)(pl)
    #do is the dropout layer, if dropout_rate is set to 0, then nothing happens here
    do = layers.Dropout(dropout_rate)(l)
    return do

#this function builds the neural network model here the number of layers is set to 1
def build_model_NN(hp):
    # define input shape
    input_shape = 97
    #inputs is the input layer, it's shape is defined in the parameter dictonary
    inputs = layers.Input(shape=(input_shape,))
    x = inputs

    #keras tuner selects the hyperparameters within the given ranges
    #dropout rate is sampled in a range of [0, 0.5]
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    #activation functions relu, tanh and gelu are considered
    activation_func = hp.Choice('activation_func', values=['relu', 'tanh', 'gelu'])
    #learning rates 0.001 and 0.01 are considered
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3])

    #number of layers is set to 1
    num_layers = 1
    #regularsizers that are considered are l1, l2, l1 and l2, no regularizer
    regularizer_choice = hp.Choice('kernel_regularizer', values=['l1', 'l2', 'l1_l2', 'None'])
    if regularizer_choice == 'l1':
        regularizer = tf.keras.regularizers.l1()
    elif regularizer_choice == 'l2':
        regularizer = tf.keras.regularizers.l2()
    elif regularizer_choice == 'l1_l2':
        regularizer = tf.keras.regularizers.l1_l2()
    else:
        regularizer = None

    # Hidden layers are set up the neurone are varied between 10 and 100 in steps of 10
    for i in range(num_layers):
        num_neurons = hp.Int(f'num_neurons_{i}', min_value=10, max_value=100, step=10)
        x = block(x, num_neurons, activation_func, regularizer, dropout_rate)

    # output is the output layer which has a linear activation function
    outputs = layers.Dense(1, activation='linear')(x)
    # here we create the model defined above, which is now called model
    model = tf.keras.Model(inputs, outputs, name='NN')

    # the optimizer is set to adam with the learning rate defined in the parameter dictionary 
    # the model is then compiled, the loss function set to the MSE and the performance metrics on the validation set during training to MAE
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model

#this function builds the neural network model here the number of layers is varied
def build_model_NN_with_layer_tuning(hp):
    # define input shape
    input_shape = 97
    #inputs is the input layer, it's shape is defined in the parameter dictonary
    inputs = layers.Input(shape=(input_shape,)) 
    x = inputs

    #keras tuner selects the hyperparameters within the given ranges
    #dropout rate is sampled in a range of [0, 0.5]
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    #activation functions relu, tanh and gelu are considered
    activation_func = hp.Choice('activation_func', values=['relu', 'tanh', 'gelu'])
    #learning rates 0.001 and 0.01 are considered
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3])
    #the number of hidden layers is varied between 1 and 3 layers
    num_layers = hp.Int('num_layers', min_value=1, max_value=3, step=1)
    #regularsizers that are considered are l1, l2, l1 and l2, no regularizer
    regularizer_choice = hp.Choice('kernel_regularizer', values=['l1', 'l2', 'l1_l2', 'None'])
    if regularizer_choice == 'l1':
        regularizer = tf.keras.regularizers.l1()
    elif regularizer_choice == 'l2':
        regularizer = tf.keras.regularizers.l2()
    elif regularizer_choice == 'l1_l2':
        regularizer = tf.keras.regularizers.l1_l2()
    else:
        regularizer = None

    # we stack as many blocks as there are hidden layers, for every hidden layer the number of variable is varied between 10 and 100 in steps of 10
    for i in range(num_layers):
        num_neurons = hp.Int(f'num_neurons_{i}', min_value=10, max_value=100, step=10)
        x = block(x, num_neurons, activation_func, regularizer, dropout_rate)

    # output is the output layer which has a linear activation function
    outputs = layers.Dense(1, activation='linear')(x)
    # here we create the model defined above, which is now called model
    model = tf.keras.Model(inputs, outputs, name='NN')

    # the optimizer is set to adam with the learning rate defined in the parameter dictionary 
    # the model is then compiled, the loss function set to the MSE and the performance metrics on the validation set during training to MAE
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model

#Performs Bayesian optimization with cross-validation to find the best hyperparameters for a model.
def bayesian_optimization_cv(build_model, X, y, cv, max_trials, executions_per_trial, patience, epochs):

    best_score = float('inf')  # Lower MSE is better
    best_model = None
    best_hp = None

    # Initialize K-Fold cross-validation
    kf = KFold(n_splits=cv)

    # Perform cross-validation
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Define the Bayesian tuner
        tuner = keras_tuner.BayesianOptimization(
            build_model,
            objective='val_mae',  # Objective for regression
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            overwrite=True,
            max_consecutive_failed_trials=1
        )

        # Early stopping callback
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

        # Perform hyperparameter tuning
        tuner.search(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[stop_early])

        # Retrieve the best hyperparameters and model from tuning
        current_best_hp = tuner.get_best_hyperparameters()[0]
        current_best_model = tuner.get_best_models(num_models=1)[0]

        # Train the best model on the current fold
        current_best_model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=0)

        # Evaluate model performance on the validation set
        val_mse = current_best_model.evaluate(X_test, y_test, verbose=0)[1]  # Get MSE

        # Update the best model if the current one performs better
        if val_mse < best_score:
            best_score = val_mse
            best_model = current_best_model
            best_hp = current_best_hp

    return best_model, best_score, best_hp

# Performs nested cross-validation using Bayesian optimization in the inner loop
def nested_cross_validation(estimator, X, y, outer_cv=5, inner_cv=5, max_trials=10, executions_per_trial=1, patience=20, epochs=100):
    outer_scores = []
    best_hyperparameters_list = []

    # Initialize outer cross-validation
    outer_kf = KFold(n_splits=outer_cv)

    # Loop through each outer fold
    for train_index, test_index in outer_kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Perform Bayesian optimization on the training set using inner CV
        best_model, best_score, best_hp = bayesian_optimization_cv(
            estimator, X_train, y_train, cv=inner_cv, max_trials=max_trials,
            executions_per_trial=executions_per_trial, patience=patience, epochs=epochs
        )

        # Evaluate the best model on the outer test set
        test_mse = best_model.evaluate(X_test, y_test, verbose=0)[1]  # Get MSE
        outer_scores.append(test_mse)
        best_hyperparameters_list.append(best_hp)

    # Calculate mean test MSE across all outer folds
    mean_outer_score = np.mean(outer_scores)

    # Select the best hyperparameters from the best-performing fold
    best_hyperparameters = best_hyperparameters_list[np.argmin(outer_scores)]

    return mean_outer_score, best_hyperparameters_list, outer_scores, best_hyperparameters
