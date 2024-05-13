import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import model_function
import keras
import tensorflow.keras.backend as K


exponential_count =2
parameter_count = exponential_count*2+1

load = True
model_path ='models/2024-01-18_18-37-14_2exp_params_205_points_50/model.h5'

input_filename ="C_4_1000_20231213_14_15_50"
df = pd.read_csv('measurements/C_4_1000_20231213_14_15_50.csv', sep=', ')
# df = pd.read_csv('measurements/E_4_20ms_20231103_14_55_46.csv', sep=', ')
df.head()

number_of_epochs = 1500

not_completed = []
predicted_param_list = []





def custom_loss(y_true, y_pred):
    """
    Custom loss function for the specified task.

    Parameters:
    - y_true: Ground truth values, of shape (batch_size, 2, N).
    - y_pred: Predicted values, of shape (batch_size, 7,N).

    Returns:
    - loss: Scalar value representing the mean loss over the batch.
    """

    # Extract x and y values from y_true
    x_values = y_true[:,:,0]  # Shape: (batch_size, N)
    y_values = y_true[:,:,1]  # Shape: (batch_size, N)
    
    function_values = tf.zeros_like(x_values)
    for i in range(int((y_pred.shape[2]-1)/2)):
        temp_val= y_pred[:, :,i]*i * K.exp(-(x_values) * y_pred[:, :,i+1]*i)
        function_values = function_values+ temp_val
    function_values = function_values+y_pred[:,:,-1]
    
    # square absolute error
    diff = (y_values -function_values) 
    squared_diff = tf.math.square(diff)
    mse = tf.math.reduce_sum(squared_diff, axis=-1)


    # diff = function_values - y_values
    # # print(diff)
    # # Take the square of the differences
    # squared_diff = K.square(diff)

    # slope_comparison = 0.0
    # for i in range(y_true.shape[2]-1):
    #     x_diff = x_values[:,i+1]-x_values[:,i]  
    #     function_slope = (function_values[:,i+1]-function_values[:,i])/x_diff 
    #     true_slope= (y_values[:,i+1]-y_values[:,i])/x_diff
    #     slope_comparison += tf.math.square(true_slope- function_slope)      

    # square relative error
    # diff = (y_values -function_values) / (y_values+0.0001) #K.maximum(y_values, y_values+0.00001)
    # squared_diff = tf.math.square(diff)
    # mse = tf.math.reduce_sum(squared_diff, axis=-1)

    loss = mse #+ slope_comparison
    # loss = K.mean(squared_diff, axis=-1)
    return loss
 

def create_nn():
    global X_train, points
    # Input layer, the number of input nodes is governed by X_data.shape[1]
    # X_data.shape[1] is the number of columns in X_data
    inputs = keras.Input(shape=(X_train.shape[1],), name='input')
    
    # Dense layers 
    #layers_norm = keras.layers.BatchNormalization()(inputs)
    layers_dense = keras.layers.Dense(10, 'linear')(inputs)
    layers_dense2 = keras.layers.Dense(10, 'linear')(layers_dense)
    # layers_dense3 = keras.layers.Dense(20, 'linear')(layers_dense2)
    # layers_dense4 = keras.layers.Dense(20, 'linear')(layers_dense3)
    # Parameter layer
    # layers_norm = keras.layers.LayerNormalization()(layers_dense2)
    parameters = keras.layers.Dense(parameter_count)(layers_dense2)
    # Expand parameters to have same shape as y_true
    expanded_parameters = keras.layers.RepeatVector(points)(parameters)


    return keras.Model(inputs=inputs, outputs=expanded_parameters, name="current_function_prediction")



def compile_model(model):
    sgd = keras.optimizers.RMSprop(clipnorm=5)
    model.compile(optimizer=sgd, #'adam'
                  loss=custom_loss)
 




earlystopper = keras.callbacks.EarlyStopping(monitor="loss",baseline = 1, patience=number_of_epochs,restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', 
                              factor=0.8, 
                              patience=5, 
                              min_lr=1e-6)




df = df[ ['Pattern','Time','Ch4(mA)','On time', 'Off time']]
df = df.rename(columns={'Ch4(mA)': 'Current'})
df_original = df.drop(df.index[:50*10])

for i in  range(int(df_original.shape[0]/50)):
    data = df_original.iloc[i*50:(i+1)*50]

    x_time = data['Time']-np.min(data['Time'])
    y_current = data['Current']
    points = len(y_current) 
    X_data = np.zeros((1,3))
    #initialize array of expected shape
    y_data = np.zeros((1, points, 2))
    
    label = np.column_stack((x_time,y_current))  
    y_data[0,:,:]= label
    features =  np.array([np.min(y_current.values), data['On time'].iloc[0], data['Off time'].iloc[0]])
    X_data[0,:] = features

    X_train = X_data
    y_train = y_data

    if load:
        model = keras.models.load_model(model_path, custom_objects={'custom_loss':custom_loss})
    else:
        model = create_nn()

    compile_model(model)

    history = model.fit(X_train, y_train,
                    batch_size=1,
                    epochs=number_of_epochs,
                    # validation_data = (X_val,y_val),
                    callbacks=[earlystopper
                               ,reduce_lr
                               ],
                    verbose=0)
    history_df = pd.DataFrame.from_dict(history.history)
    final_loss = history_df['loss'].iloc[-1]
    if final_loss < 0.1:
        y_train_prediction = model.predict(X_train)
        param_predicted = y_train_prediction[0,0 ,: ]
        predicted_param_list.append(param_predicted)
        # print(param_predicted)
    
    else:
        predicted_param_list.append(np.zeros_like(param_predicted))

    #print progress    
    print(i,'/', int(df_original.shape[0]/50))




df_params = pd.DataFrame(predicted_param_list)

# Save the DataFrame to a CSV file
df_params.to_csv('outputs/'+input_filename+'_params.csv', index=False, header=False)