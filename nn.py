import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def prepare_data(df, input_cols,output_var,test_size=0.3,scaley=False):
    """
    """
    df1 = df.copy()
    X = df1[input_cols]
    y = df1[str(output_var)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Min-max scaler from Scikit-learn
    scalerx = MinMaxScaler()
    scalery = MinMaxScaler()
    X_train_scaled = scalerx.fit_transform(X_train)
    X_test_scaled = scalerx.fit_transform(X_test)
    if scaley == True:
        y_train_scaled = scalery.fit_transform(y_train.values.reshape(-1, 1))
        y_test_scaled = scalery.fit_transform(y_test.values.reshape(-1, 1))
    else:
        y_train_scaled = y_train.values.reshape(-1, 1)
        y_test_scaled = y_test.values.reshape(-1, 1)
    
    return (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled)


def build_model(num_layers=1, architecture=[32],act_func='relu', 
                input_dim=2, output_class=10):
  """
  Builds a densely connected neural network model from user input
  num_layers: Number of hidden layers
  architecture: Architecture of the hidden layers (densely connected)
  act_func: Activation function. Could be 'relu', 'sigmoid', or 'tanh'.
  input_shape: Dimension of the input vector
  output_class: Number of classes in the output vector
  """
  layers=[tf.keras.layers.Dense(input_dim,input_dim=input_dim)]
  if act_func=='relu':
    activation=tf.nn.relu
  elif act_func=='sigmoid':
    activation=tf.nn.sigmoid
  elif act_func=='tanh':
    activation=tf.nn.tanh
    
  for i in range(num_layers):
    layers.append(tf.keras.layers.Dense(architecture[i], activation=tf.nn.relu))
  layers.append(tf.keras.layers.Dense(1))
  
  model = tf.keras.models.Sequential(layers)
  return model


def compile_train_model(model,x_train, y_train, callbacks=None,
                        learning_rate=0.001,batch_size=1,epochs=10,verbose=0):
  """
  Compiles and trains a given Keras model with the given data. 
  Assumes Adam optimizer for this implementation.
  
  learning_rate: Learning rate for the optimizer Adam
  batch_size: Batch size for the mini-batch optimization
  epochs: Number of epochs to train
  verbose: Verbosity of the training process
  """
  
  model_copy = model
  model_copy.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                     loss="mse", metrics=["mse"])
  
  if callbacks is not None:
        model_copy.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                 callbacks=[callbacks],verbose=verbose)
  else:
    model_copy.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                   verbose=verbose)
  return model_copy