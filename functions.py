import numpy as np
import matplotlib.pyplot as plt

def pixel_to_signal_array(input:np.ndarray, base_on_time:int, on_time:int, off_time:int, modulation:float, max_int:int, linearize = False):
    global model, baseline_current
    if linearize:
        input = input.reshape(-1, input.shape[-1]**2)
    #normalize input to range [0,1]
    input = input.T/max_int
    
    #create output arrays of proper size
    output_current = np.zeros_like(input)
    
    #create auxiliary arrays to hold on/off times
    on_array =  np.zeros_like(input)
    off_array = np.zeros_like(input)
    

  
    
    #create auxiliary array with baseline current
    baseline_current_array = np.empty_like(input)
    baseline_current_array.fill(baseline_current)
    
    # set first on off time based on first column input
    for i, input_row in enumerate(input):
        on_array[i] = (base_on_time+on_time*(input_row*modulation)).astype(int)
        off_array[i] = (off_time+on_time*((1-input_row)*modulation)).astype(int)
        
        if i ==0: 
            model_input= np.stack([baseline_current_array[0,:], on_array[i], off_array[i]], axis=1)
            # print(i, model_input)
            # print(model.predict(model_input, verbose=0).flatten()) 
            # print(output_current[i,:]  )
            output_current[i,:] = model.predict(model_input, verbose=0).flatten()
            # print(output_current[i,:])
        else:
            model_input= np.stack([output_current[i-1], on_array[i], off_array[i]], axis=1)
            output_current[i,:] = model.predict(model_input, verbose=0).flatten()
        print(f'{i}/{input.shape[0]}', end='\r')      
             

    # Transpose back           
    return output_current.T, on_array.T, off_array.T

def draw_pulse_train_array(source_array,current,on_times,off_times, linearize = False):
    pulse_train = np.zeros((current.shape[0],off_times.shape[1]*2))
    # pulse_train = pulse_train.T
    pulse_train[:, ::2] = on_times
    pulse_train[:, 1::2] = off_times
    # pulse_train = pulse_train.T
    fig, axs = plt.subplots(pulse_train.shape[0],sharex= True, figsize=(10,3))
    for irow, row in enumerate(pulse_train):
        y_values = []
        for i,value in enumerate(row):
            if i%2==0:
                y_values.extend([1]*int(value))
            else:
                y_values.extend([0]*int(value))
        if linearize:
            axs.fill_between(range(len(y_values)),y_values, linestyle='-', label=irow)      
        else:
            axs[irow].fill_between(range(len(y_values)),y_values, linestyle='-', label=irow)  
        # plt.legend()
    plt.xlabel('Time (ms)')
    fig.supylabel('Signal')
    fig.suptitle('Device input signal per row')


    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(3,3))
    axs.imshow(source_array, cmap='gray')
    axs.set_xlabel('Column count')
    axs.set_ylabel('Row count')
    axs.set_title('Source image')

    
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(3,3))
    axs.imshow(current, cmap='inferno', aspect='auto')
    axs.set_xlabel('Pulse count')
    axs.set_ylabel('Row count')
    axs.set_title('Current heatmap')

 
    plt.figure(figsize=(10, 3))
    for i, row in enumerate(current):
        plt.plot(range(len(row)),row, marker='.', linestyle='-', label=i)  
    # plt.legend()
    plt.xlabel('Pulse count')
    plt.ylabel('Current (mA)')
    plt.title('Device output per row signal')

    
