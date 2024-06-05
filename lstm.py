import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

def train_lstm_model(xtrain, ytrain, time_steps, task_type="classification"):
    """
    Trains an LSTM model on the given training data.
    
    Parameters:
    xtrain (pd.DataFrame): Training data input.
    ytrain (pd.DataFrame or pd.Series): Training data labels.
    time_steps (int): The number of time steps in your data sequence.
    task_type (str): Type of the task, either "classification" or "regression".
    
    Returns:
    model: A trained LSTM model.
    
    Explanation:
    # This script provides flexibility for handling both classification and regression tasks. You simply need to specify
    # the `task_type` when calling the functions. The `num_features` is inferred from the width of `xtrain` divided by the number of time steps, assuming each time step has the same number of features.
    # Remember, for time series forecasting, your data should be structured such that each row represents a time step, 
    # and each column (except the target column) represents a feature. The `time_steps` parameter indicates how many previous time steps are used to predict the next step. Adjust the `time_steps` value based on your specific problem and data.
    # Also, note that for classification tasks, the script assumes that your `ytrain` and `ytest` contain integer class 
    # labels. If your labels are already in one-hot encoded format, you should modify the script accordingly to avoid re-encoding them.
    # For regression tasks, ensure that your target variable is a continuous value. The script is designed to handle a 
    # single target variable for simplicity, but it can be adapted for multi-target regression by adjusting the output 
    # layer of the LSTM model.
    
    Example:
    # Example usage
    # Assuming you have your dataframe `df` and have already split it into `xtrain`, `xtest`, `ytrain`, `ytest`
    # and have determined the necessary parameter: `time_steps`.

    # Train the model
    # For classification:
    model = train_lstm_model(xtrain, ytrain, time_steps=your_time_steps, task_type="classification")

    # For regression:
    # model = train_lstm_model(xtrain, ytrain, time_steps=your_time_steps, task_type="regression")

    # Make predictions
    # For classification:
    predictions = predict_with_lstm(model, xtest, time_steps=your_time_steps, task_type="classification")

    # For regression:
    # predictions = predict_with_lstm(model, xtest, time_steps=your_time_steps, task_type="regression")

    """
    # Convert DataFrames to numpy arrays
    xtrain_array = xtrain.values
    ytrain_array = ytrain.values
    
    # Infer number of features
    num_features = xtrain_array.shape[1] // time_steps
    
    # Reshape data for LSTM [samples, time_steps, features]
    xtrain_reshaped = xtrain_array.reshape((-1, time_steps, num_features))
    
    # Prepare labels based on task type
    if task_type == "classification":
        # Assuming ytrain contains class labels, convert to one-hot encoding
        num_classes = len(np.unique(ytrain_array))
        ytrain_prepared = to_categorical(ytrain_array, num_classes=num_classes)
        loss_function = 'categorical_crossentropy'
        output_activation = 'softmax'
        output_dim = num_classes
    elif task_type == "regression":
        # For regression tasks, no change needed
        ytrain_prepared = ytrain_array
        loss_function = 'mean_squared_error'
        output_activation = 'linear'
        output_dim = 1
    else:
        raise ValueError("task_type must be either 'classification' or 'regression'")
    
    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(100, input_shape=(time_steps, num_features)))
    model.add(Dense(output_dim, activation=output_activation))
    
    # Compile the model
    model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy' if task_type == "classification" else 'mse'])
    
    # Fit the model to the training data
    model.fit(xtrain_reshaped, ytrain_prepared, epochs=10, batch_size=32, verbose=1)
    
    return model


def predict_with_lstm(model, xtest, time_steps, task_type="classification"):
    """
    Makes predictions on test data using the provided trained LSTM model.
    
    Parameters:
    model: A trained LSTM model.
    xtest (pd.DataFrame): Test data input.
    time_steps (int): The number of time steps in your data sequence.
    task_type (str): Type of the task, either "classification" or "regression".
    
    Returns:
    predictions: Predictions made by the LSTM model.
    """
    # Convert DataFrame to numpy array
    xtest_array = xtest.values
    
    # Infer number of features
    num_features = xtest_array.shape[1] // time_steps
    
    # Reshape data for LSTM [samples, time_steps, features]
    
    xtest_reshaped = xtest_array.reshape((-1, time_steps, num_features))
    
    # Make predictions
    predictions = model.predict(xtest_reshaped)
    
    # Post-process predictions based on task type
    if task_type == "classification":
        # For classification, convert predicted probabilities to class labels
        predictions = np.argmax(predictions, axis=1)
    elif task_type == "regression":
        # For regression, no post-processing needed if single output
        pass
    
    return predictions




import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, LeakyReLU, Activation
from tensorflow.keras.utils import to_categorical

# Custom activation functions
def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

def swish(x):
    return x * tf.sigmoid(x)

# Dictionary of activation functions
activation_functions = {
    'relu': tf.keras.activations.relu,
    'leaky_relu': tf.keras.layers.LeakyReLU(),
    'gelu': tf.keras.activations.gelu,
    'mish': mish,
    'swish': swish
}

def train_stacked_lstm_model(xtrain, ytrain, time_steps, task_type="classification", num_layers=2, activation='relu'):
    """
    Trains a stacked LSTM model on the given training data.
    
    Parameters:
    xtrain (pd.DataFrame): Training data input.
    ytrain (pd.DataFrame or pd.Series): Training data labels.
    time_steps (int): The number of time steps in your data sequence.
    task_type (str): Type of the task, either "classification" or "regression".
    num_layers (int): Number of LSTM layers in the stacked model.
    activation (str): Activation function to use in LSTM layers.
    
    Returns:
    model: A trained stacked LSTM model.
    
    Explanation:
    A stacked LSTM model, with parameters to control the number of LSTM layers and the activation 
    function used in the LSTM layers. Also includes options for ReLU, Leaky ReLU, GELU, Mish, and Swish. Note that 
    LSTM gates traditionally use the `tanh` activation function, but the recurrent connections (or the cell state 
    update) can benefit from these modern activation functions. Also added are the `tf.keras.layers.LeakyReLU`, 
    `tf.keras.activations.gelu`, `tf.keras.activations.swish`, and custom Mish and Swish activation functions as needed.

    Example:
    # Example usage
    # Assuming you have your dataframe `df` and have already split it into `xtrain`, `xtest`, `ytrain`, `ytest`
    # and have determined the necessary parameter: `time_steps`.

    # Train the model
    # For classification with 3 stacked LSTM layers using Leaky ReLU activation:
    model = train_stacked_lstm_model(xtrain, ytrain, time_steps=your_time_steps, task_type="classification", num_layers=3, activation="leaky_relu")

    # For regression with 2 stacked LSTM layers using GELU activation:
    # model = train_stacked_lstm_model(xtrain, ytrain, time_steps=your_time_steps, task_type="regression", num_layers=2, activation="gelu")

    """
    # Convert DataFrames to numpy arrays
    xtrain_array = xtrain.values
    ytrain_array = ytrain.values
    
    # Infer number of features
    num_features = xtrain_array.shape[1] // time_steps
    
    # Reshape data for LSTM [samples, time_steps, features]
    xtrain_reshaped = xtrain_array.reshape((-1, time_steps, num_features))
    
    # Prepare labels based on task type
    if task_type == "classification":
        # Assuming ytrain contains class labels, convert to one-hot encoding
        num_classes = len(np.unique(ytrain_array))
        ytrain_prepared = to_categorical(ytrain_array, num_classes=num_classes)
        loss_function = 'categorical_crossentropy'
        output_activation = 'softmax'
        output_dim = num_classes
    elif task_type == "regression":
        # For regression tasks, no change needed
        ytrain_prepared = ytrain_array
        loss_function = 'mean_squared_error'
        output_activation = 'linear'
        output_dim = 1
    else:
        raise ValueError("task_type must be either 'classification' or 'regression'")
    
    # Define the stacked LSTM model
    model = Sequential()
    
    # Add the first LSTM layer
    model.add(LSTM(100, activation=activation_functions.get(activation, 'tanh'), return_sequences=True, input_shape=(time_steps, num_features)))
    
    # Add additional LSTM layers based on num_layers
    for _ in range(num_layers - 1):
        model.add(LSTM(100, activation=activation_functions.get(activation, 'tanh, return_sequences=False if _ == num_layers - 2 else True))
    
    # Add the output layer
    model.add(Dense(output_dim, activation=output_activation))
    
    # Compile the model
    model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy' if task_type == "classification" else 'mse'])
    
    # Fit the model to the training data
    model.fit(xtrain_reshaped, ytrain_prepared, epochs=10, batch_size=32, verbose=1)
    
    return model


def predict_with_stacked_lstm(model, xtest, time_steps, task_type="classification"):
    """
    Makes predictions on test data using the provided trained stacked LSTM model.
    
    Parameters:
    model: A trained stacked LSTM model.
    xtest (pd.DataFrame): Test data input.
    time_steps (int): The number of time steps in your data sequence.
    task_type (str): Type of the task, either "classification" or "regression".
    
    Returns:
    predictions: Predictions made by the stacked LSTM model.
    
    Explanation:
    A convenience wrapper for predicting results from the stacked LSTM model.
    
    Example:
    # Make predictions
    # For classification:
    predictions = predict_with_stacked_lstm(model, xtest, time_steps=your_time_steps, task_type="classification")

    # For regression:
    # predictions = predict_with_stacked_lstm(model, xtest, time_steps=your_time_steps, task_type="regression")

    """
    # Convert DataFrame to numpy array
    xtest_array = xtest.values
    
    # Infer number of features
    num_features = xtest_array.shape[1] // time_steps
    
    # Reshape data for LSTM [samples, time_steps, features]
    xtest_reshaped = xtest_array.reshape((-1, time_steps, num_features))
    
    # Make predictions
    predictions = model.predict(xtest_reshaped)
    
    # Post-process predictions based on task type
    if task_type == "classification":
        # For classification, convert predicted probabilities to class labels
        predictions = np.argmax(predictions, axis=1)
    elif task_type == "regression":
        # For regression, no post-processing needed if single output
        pass
    
    return predictions



