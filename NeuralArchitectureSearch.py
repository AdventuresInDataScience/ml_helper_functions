import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Union, Tuple
from kerastuner.tuners import RandomSearch, Hyperband, BayesianOptimization
from kerastuner.engine.hyperparameters import HyperParameters

# Custom activation functions if not standard in Keras
def gelu(x):
    """
    Gaussian Error Linear Unit activation function.
    """
    return 0.5 * x * (1 + tf.math.tanh(tf.math.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

def mish(x):
    """
    Mish activation function.
    """
    return x * tf.math.tanh(tf.math.softplus(x))

def swish(x):
    """
    Swish activation function.
    """
    return x * tf.nn.sigmoid(x)

# Register custom activations in Keras
tf.keras.utils.get_custom_objects().update({'gelu': tf.keras.layers.Activation(gelu)})
tf.keras.utils.get_custom_objects().update({'mish': tf.keras.layers.Activation(mish)})
tf.keras.utils.get_custom_objects().update({'swish': tf.keras.layers.Activation(swish)})

def build_model(hp: HyperParameters, 
                input_shape: Tuple[int], 
                output_units: int, 
                problem_type: str, 
                objective: str) -> tf.keras.Model:
    """
    Builds a neural network model based on hyperparameters.

    Parameters:
    hp (HyperParameters): Hyperparameters for the model.
    input_shape (Tuple[int]): Shape of the input data.
    output_units (int): Number of output units.
    problem_type (str): Type of problem - 'classification' or 'regression'.
    objective (str): Objective function for the model.

    Returns:
    tf.keras.Model: Compiled Keras model.
    """
    model = tf.keras.Sequential()
    
    for i in range(hp.Int('num_layers', 1, 5)):
        layer_type = hp.Choice('layer_type_' + str(i), ['dense', 'cnn', 'rnn', 'lstm'])
        if layer_type == 'dense':
            model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
                                            activation=hp.Choice('activation_' + str(i), ['gelu', 'mish', 'swish', 'relu', 'leaky_relu', 'tanh', 'sigmoid'])))
        elif layer_type == 'cnn':
            if len(input_shape) == 1:
                input_shape = (input_shape[0], 1, 1)
            if len(model.layers) == 0:
                model.add(tf.keras.layers.Reshape((input_shape[0], input_shape[1], 1), input_shape=input_shape))
            model.add(tf.keras.layers.Conv2D(filters=hp.Int('filters_' + str(i), min_value=32, max_value=128, step=32),
                                             kernel_size=hp.Choice('kernel_size_' + str(i), [3, 5]),
                                             activation=hp.Choice('activation_' + str(i), ['gelu', 'mish', 'swish', 'relu', 'leaky_relu', 'tanh', 'sigmoid'])))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        elif layer_type == 'rnn':
            if len(input_shape) == 1:
                input_shape = (input_shape[0], 1)
            if len(model.layers) == 0:
                model.add(tf.keras.layers.Reshape((input_shape[0], input_shape[1]), input_shape=input_shape))
            model.add(tf.keras.layers.SimpleRNN(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
                                                activation=hp.Choice('activation_' + str(i), ['gelu', 'mish', 'swish', 'relu', 'leaky_relu', 'tanh', 'sigmoid']),
                                                return_sequences=True if i < hp.get('num_layers') - 1 else False))
        elif layer_type == 'lstm':
            if len(input_shape) == 1:
                input_shape = (input_shape[0], 1)
            if len(model.layers) == 0:
                model.add(tf.keras.layers.Reshape((input_shape[0], input_shape[1]), input_shape=input_shape))
            model.add(tf.keras.layers.LSTM(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
                                           activation=hp.Choice('activation_' + str(i), ['gelu', 'mish', 'swish', 'relu', 'leaky_relu', 'tanh', 'sigmoid']),
                                           return_sequences=True if i < hp.get('num_layers') - 1 else False))
    
    if problem_type == 'classification':
        model.add(tf.keras.layers.Dense(output_units, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        model.add(tf.keras.layers.Dense(output_units))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

def neural_network_search(Xtrain: Union[pd.DataFrame, pd.Series, np.ndarray],
                          ytrain: Union[pd.DataFrame, pd.Series, np.ndarray],
                          problem_type: str,
                          tuning_method: str = 'random_search',
                          max_trials: int = 10,
                          executions_per_trial: int = 1,
                          epochs: int = 10,
                          early_stopping_patience: int = 5,
                          min_layers: int = 1,
                          max_layers: int = 5,
                          min_units: int = 32,
                          max_units: int = 512,
                          unit_step: int = 32,
                          objective: str = 'infer',
                          **kwargs) -> Tuple[tf.keras.Model, HyperParameters]:
    """
    Perform a neural network architecture search and fit the best model to the data.

    Parameters:
    Xtrain (Union[pd.DataFrame, pd.Series, np.ndarray]): Training features
    ytrain (Union[pd.DataFrame, pd.Series, np.ndarray]): Training labels
    problem_type (str): Type of problem - 'classification' or 'regression'
    tuning_method (str): Method to use for tuning - 'random_search', 'hyperband', or 'bayesian'
    max_trials (int): Maximum number of trials for hyperparameter tuning
    executions_per_trial (int): Number of models to be built and fit for each trial
    epochs (int): Number of epochs to train each model
    early_stopping_patience (int): Number of epochs with no improvement after which training will be stopped
    min_layers (int): Minimum number of layers to consider
    max_layers (int): Maximum number of layers to consider
    min_units (int): Minimum number of units for dense layers
    max_units (int): Maximum number of units for dense layers
    unit_step (int): Step size for number of units in dense layers
    objective (str): Objective function to use for tuning - 'infer' to automatically infer based on problem type
    **kwargs: Additional arguments for tuning and model fitting

    Returns:
    Tuple[tf.keras.Model, HyperParameters]: Fitted model and final model specification

    Example:
    model, best_hp = neural_network_search(
        Xtrain, ytrain, 
        problem_type='classification', 
        tuning_method='random_search', 
        max_trials=10, 
        executions_per_trial=2, 
        epochs=50, 
        early_stopping_patience=5,
        min_layers=1,
        max_layers=5,
        min_units=32,
        max_units=512,
        unit_step=32,
        objective='infer'
    )
    """
    # Validate inputs
    if problem_type not in ['classification', 'regression']:
        raise ValueError("problem_type must be 'classification' or 'regression'")
    
    if not isinstance(Xtrain, (pd.DataFrame, pd.Series, np.ndarray)):
        raise TypeError("Xtrain must be a pandas DataFrame, Series, or numpy array")
    
    if not isinstance(ytrain, (pd.DataFrame, pd.Series, np.ndarray)):
        raise TypeError("ytrain must be a pandas DataFrame, Series, or numpy array")
    
    # Convert Series to DataFrame if necessary
    if isinstance(Xtrain, pd.Series):
        Xtrain = Xtrain.to_frame()
    
    if isinstance(ytrain, pd.Series):
        ytrain = ytrain.to_frame()
    
    # Convert inputs to numpy arrays if they are DataFrames
    if isinstance(Xtrain, pd.DataFrame):
        Xtrain = Xtrain.values
    
    if isinstance(ytrain, pd.DataFrame):
        ytrain = ytrain.values

    # Ensure ytrain is a flat array for classification problems
    if problem_type == 'classification':
        ytrain = ytrain.ravel()

    # Determine input shape and output units
    input_shape = Xtrain.shape[1:]
    output_units = len(np.unique(ytrain)) if problem_type == 'classification' else 1
    
    # Infer objective function if set to 'infer'
    if objective == 'infer':
        objective = 'val_accuracy' if problem_type == 'classification' else 'val_loss'
    
    # Validate objective function
    valid_objectives = ['val_accuracy', 'val_loss', 'accuracy', 'loss', 'mse', 'mae', 'sparse_categorical_crossentropy']


    if objective not in valid_objectives:
        raise ValueError(f"Invalid objective function '{objective}'. Must be one of {valid_objectives}.")
    
    # Set up the tuner
    hp = HyperParameters()
    
    if tuning_method == 'random_search':
        tuner = RandomSearch(
            lambda hp: build_model(hp, input_shape, output_units, problem_type, objective),
            objective=objective,
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            **kwargs
        )
    elif tuning_method == 'hyperband':
        tuner = Hyperband(
            lambda hp: build_model(hp, input_shape, output_units, problem_type, objective),
            objective=objective,
            max_epochs=epochs,
            executions_per_trial=executions_per_trial,
            **kwargs
        )
    elif tuning_method == 'bayesian':
        tuner = BayesianOptimization(
            lambda hp: build_model(hp, input_shape, output_units, problem_type, objective),
            objective=objective,
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            **kwargs
        )
    else:
        raise ValueError("tuning_method must be 'random_search', 'hyperband', or 'bayesian'")

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss' if problem_type == 'regression' else 'val_accuracy', 
                                                      patience=early_stopping_patience, 
                                                      restore_best_weights=True)

    tuner.search(Xtrain, ytrain, epochs=epochs, validation_split=0.2, callbacks=[early_stopping])
    
    # Get the optimal hyperparameters
    best_hp = tuner.get_best_hyperparameters()[0]
    
    # Build the model with the optimal hyperparameters
    model = build_model(best_hp, input_shape, output_units, problem_type, objective)
    
    # Train the model
    model.fit(Xtrain, ytrain, epochs=epochs, validation_split=0.2, callbacks=[early_stopping])
    
    return model, best_hp

# Example usage:
# model, best_hp = neural_network_search(Xtrain, ytrain, 'classification', tuning_method='random_search', max_trials=10, executions_per_trial=2, epochs=50, early_stopping_patience=5)

'''
# Example
import pandas as pd
import numpy as np

# Generate sample data
np.random.seed(0)
Xtrain = pd.DataFrame(np.random.rand(100, 10))  # 100 samples, 10 features
ytrain = pd.Series(np.random.randint(0, 2, 100))  # Binary classification labels

# Example usage with additional keyword arguments
model, best_hp = neural_network_search(
    Xtrain, ytrain, 
    problem_type='classification', 
    tuning_method='random_search', 
    max_trials=10, 
    executions_per_trial=2, 
    epochs=50, 
    early_stopping_patience=5,
    min_layers=1,
    max_layers=5,
    min_units=32,
    max_units=512,
    unit_step=32,
    objective='infer',  # Automatically infer the objective based on problem type
    # Additional keyword arguments
    overwrite=True,  # Example of additional argument
    factor=0.2,
    seed=42
)

In this example:
- We generate sample data `Xtrain` and `ytrain`.
- We call the `neural_network_search` function with various keyword arguments such as `overwrite`, `factor`, and `seed`.
- These additional arguments are passed to the function and can be used for further customization or tuning of the neural network architecture search.
'''