import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics as skm
from sklearn.metrics import PredictionErrorDisplay
#pseudocode for neural network training

# choose framework for neural network
#build own: https://realpython.com/python-ai-neural-network/
#neural network: https://keras.io/examples/vision/mlp_image_classification/
#linear regression: https://www.tensorflow.org/tutorials/keras/regression

# decide number of layers
# decide input vector of parameters for neural network first layer
#------>>>>>FEATURE SCALING!<<<<<<------

# score function:
#use keras with custom loss function https://keras.io/api/losses/#creating-custom-losses
#https://stackoverflow.com/questions/45961428/make-a-custom-loss-function-in-keras
#    use sklearn functions as loss functions: https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
        # take outputs of last layer corresponding to number of parameters of model function

        # calculate model function for given time range of label

        # compare to label by calculating:

        # -residuals: 
        # sklearn.metrics.r2_score compare each value to mean and based on te variation of individual R2 to R2_mean give those negative score 

        # -rsquare: sklearn.metrics.r2_score

        # -mean_absolute_percentage_error

        # -Max_error

display = PredictionErrorDisplay(y_true= np.array([ 1,2,3,3,6,1,2]), y_pred=np.array([0,1,4,5,6,2,3]))
display.plot()

plt.show()